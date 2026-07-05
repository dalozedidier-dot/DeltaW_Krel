"""Extended tests for the K_CS robustness SDP wiring."""
from __future__ import annotations

import numpy as np
import pytest

cvxpy = pytest.importorskip("cvxpy")

import deltawkrel.sdp as sdp_module
from deltawkrel.projectors import ProcessDims, white_noise_process
from deltawkrel.sdp import SdpDiagnostics, solve_cs_robustness, solve_cs_robustness_scaffold
from deltawkrel.switch_models import causally_separable_mixture

DIMS = ProcessDims(2, 2, 2, 2)


def test_wrong_shape_raises_value_error():
    with pytest.raises(ValueError):
        solve_cs_robustness(np.eye(4), dims=DIMS)


def test_missing_cvxpy_raises_runtime_error(monkeypatch):
    monkeypatch.setattr(sdp_module, "cp", None)
    with pytest.raises(RuntimeError):
        sdp_module.solve_cs_robustness(np.eye(16), dims=DIMS)


def test_negative_white_noise_needs_exact_robustness():
    """W0 = -0.5 W_white forces t_white + Σt_i = 0.5 exactly.

    The white-noise process lies in both fixed-order subspaces and is PSD, so
    the equality W0 + t·W_white = W_AB + W_BA is feasible iff the total white
    coefficient is >= 0.5, making the optimum analytically known.
    """
    W_white = white_noise_process(DIMS)
    diag = solve_cs_robustness(-0.5 * W_white, dims=DIMS, solver="CLARABEL")
    assert diag.status in {"optimal", "optimal_inaccurate"}
    assert diag.objective_value == pytest.approx(0.5, abs=1e-5)
    assert diag.equality_residual_fro < 1e-5


def test_calibrated_noise_direction_is_used():
    """With a calibrated noise equal to the white-noise process, the solver can
    split the required 0.5 between t_white and t_cal; only the total is fixed."""
    W_white = white_noise_process(DIMS)
    diag = solve_cs_robustness(
        -0.5 * W_white,
        dims=DIMS,
        noises=[W_white],
        solver="CLARABEL",
    )
    assert diag.status in {"optimal", "optimal_inaccurate"}
    assert diag.t_white + diag.t_calibrated_sum == pytest.approx(0.5, abs=1e-5)
    assert 0.0 <= diag.omega_white <= 1.0


def test_mixture_target_has_zero_robustness_and_zero_omega():
    diag = solve_cs_robustness(causally_separable_mixture(DIMS, q=0.7), dims=DIMS)
    assert diag.status in {"optimal", "optimal_inaccurate"}
    assert diag.objective_value < 1e-5
    assert diag.omega_white == pytest.approx(0.0, abs=1e-6)
    assert diag.equality_residual_fro < 1e-5


def test_diagnostics_serialization_round_trip():
    diag = solve_cs_robustness(white_noise_process(DIMS), dims=DIMS)
    payload = diag.to_dict()
    assert isinstance(payload, dict)
    expected_keys = {
        "status",
        "objective_value",
        "t_white",
        "t_calibrated_sum",
        "omega_white",
        "solver",
        "equality_residual_fro",
        "note",
    }
    assert set(payload.keys()) == expected_keys
    rebuilt = SdpDiagnostics(**payload)
    assert rebuilt.status == diag.status


def test_backward_compatible_scaffold_alias():
    diag = solve_cs_robustness_scaffold(white_noise_process(DIMS), dims=DIMS)
    assert diag.status in {"optimal", "optimal_inaccurate"}
    assert diag.objective_value < 1e-5


def test_unknown_solver_falls_back_to_scs(capsys):
    diag = solve_cs_robustness(white_noise_process(DIMS), dims=DIMS, solver="NO_SUCH_SOLVER")
    assert diag.solver == "SCS"
    assert diag.status in {"optimal", "optimal_inaccurate"}
    assert "fallback" in capsys.readouterr().out


def test_strict_float_maps_non_finite_to_none():
    assert sdp_module._strict_float(1.5) == 1.5
    assert sdp_module._strict_float(float("nan")) is None
    assert sdp_module._strict_float(float("inf")) is None


def test_custom_white_noise_operator_is_accepted():
    W_white = white_noise_process(DIMS)
    diag = solve_cs_robustness(-0.25 * W_white, dims=DIMS, W_white=2.0 * W_white)
    assert diag.status in {"optimal", "optimal_inaccurate"}
    # Doubling the white operator halves the required coefficient.
    assert diag.objective_value == pytest.approx(0.125, abs=1e-5)
