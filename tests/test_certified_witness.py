"""Tests for the certified single-direction causal witness on the switch."""
from __future__ import annotations

import numpy as np
import pytest

cp = pytest.importorskip("cvxpy")

from deltawkrel.certified_witness import (
    admissible_direction,
    affine_witness_curve,
    switch_generalized_robustness_witness,
)
from deltawkrel.switch_models import (
    dephased_switch_process,
    ideal_quantum_switch_process,
    partially_dephased_switch_process,
    switch_white_noise_process,
)


@pytest.fixture(scope="module")
def certificate():
    return switch_generalized_robustness_witness(ideal_quantum_switch_process(), eps=1e-9)


def test_witness_reproduces_benchmark_and_is_tight(certificate):
    assert certificate.status in {"optimal", "optimal_inaccurate"}
    assert abs(certificate.R_g - 0.5454) < 2e-3
    # strong duality: Tr(S* W_ref) == R_g
    assert certificate.tightness_gap < 1e-6


def test_witness_signal_is_affine_in_lambda(certificate):
    curve = affine_witness_curve(
        certificate.S, partially_dephased_switch_process, np.linspace(0, 1, 11)
    )
    assert curve.affinity_residual < 1e-9
    # tight at the reference process
    assert abs(curve.values[0] - certificate.R_g) < 1e-6
    # decreasing toward the causally separable endpoint
    assert curve.slope < 0
    assert 0.0 < curve.zero_crossing < 1.0


def test_single_scalar_inversion_recovers_lambda(certificate):
    curve = affine_witness_curve(certificate.S, partially_dephased_switch_process)
    for lam in (0.15, 0.42, 0.83):
        y = certificate.value(partially_dephased_switch_process(lam))
        assert abs(curve.invert(y) - lam) < 1e-9


def test_krel_is_immune_to_calibrated_noise(certificate):
    D = certificate.S.shape[0]
    calibrated = [np.eye(D), switch_white_noise_process() - ideal_quantum_switch_process()]
    signal = dephased_switch_process() - ideal_quantum_switch_process()
    adm = admissible_direction(certificate.S, calibrated, signal)
    for leak in adm.noise_leakage.values():
        assert abs(leak) < 1e-9          # first-order blind to nuisance directions
    assert adm.signal_response != 0.0    # still detects the falsification axis
    assert 0.0 < adm.retained_fraction <= 1.0
