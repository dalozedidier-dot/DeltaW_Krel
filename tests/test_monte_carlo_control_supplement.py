"""Smoke and invariant tests for the linearized ΔW/K_rel control script.

These tests validate the toy/geometric level only. They do not validate the full
causal SDP on CS or full process-matrix tomography.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "monte_carlo_control_supplement.py"
spec = importlib.util.spec_from_file_location("monte_carlo_control_supplement", SCRIPT_PATH)
mc = importlib.util.module_from_spec(spec)
assert spec.loader is not None
sys.modules[spec.name] = mc
spec.loader.exec_module(mc)


def test_symmetrize_and_traceless_invariants() -> None:
    rng = np.random.default_rng(1)
    matrix = rng.normal(size=(5, 5))
    sym = mc.symmetrize(matrix)
    assert np.allclose(sym, sym.T)
    tl = mc.traceless(sym)
    assert abs(float(np.trace(tl))) < 1e-10


def test_basis_control_builds_applicable_direction() -> None:
    rng = np.random.default_rng(2)
    basis = mc.build_basis_control(dim=4, rng=rng, alpha_true=0.12, n_noise=1, eps_num=1e-10)
    diagnostics = mc.basis_diagnostics(basis)
    assert basis.A_dim > 0
    assert basis.projection_norm > basis.eps_num
    assert np.isclose(np.linalg.norm(basis.K_rel), 1.0)
    assert diagnostics["max_inner_N_K_rel"] < 1e-8


def test_applicability_lock_rejects_empty_or_too_small_projection() -> None:
    with pytest.raises(mc.InapplicableConfiguration):
        mc.apply_applicability_lock([], np.eye(2), eps_num=1e-8)
    rng = np.random.default_rng(3)
    basis = mc.build_basis_control(dim=4, rng=rng, alpha_true=0.12, n_noise=1, eps_num=1e-12)
    with pytest.raises(mc.InapplicableConfiguration):
        mc.apply_applicability_lock(basis.A_basis, basis.S_opt, eps_num=10.0)


def test_monte_carlo_smoke_run_without_outputs() -> None:
    rows = mc.run_monte_carlo_control(
        lambda_true_values=[0.0, 0.005],
        n_samples_values=[50],
        n_sim=10,
        n_null=20,
        dim=4,
        n_noise=1,
        seed_basis=11,
        seed_sim=12,
        save_outputs=False,
        make_plots=False,
    )
    assert len(rows) == 2
    assert {row.lambda_true for row in rows} == {0.0, 0.005}
    assert all(0.0 <= row.power <= 1.0 for row in rows)
