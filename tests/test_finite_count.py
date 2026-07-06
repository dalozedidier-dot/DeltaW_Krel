"""Tests for the A4 finite-count single-direction estimator."""
from __future__ import annotations

import numpy as np
import pytest

cp = pytest.importorskip("cvxpy")

from deltawkrel.certified_witness import admissible_direction, switch_generalized_robustness_witness
from deltawkrel.finite_count import (
    copies_to_certify,
    false_positive_under_drift,
    lambda_estimator_scaling,
    per_copy_variance,
)
from deltawkrel.switch_models import (
    dephased_switch_process,
    ideal_quantum_switch_process,
    partially_dephased_switch_process,
)


@pytest.fixture(scope="module")
def witness():
    return switch_generalized_robustness_witness(ideal_quantum_switch_process(), eps=1e-9).S


def test_variance_scales_as_one_over_N(witness):
    res = lambda_estimator_scaling(witness, partially_dephased_switch_process,
                                   lambda_ref=0.1, n_repeat=300)
    ratio = res.var_lambda_empirical / res.var_lambda_analytic
    assert np.all(np.abs(ratio - 1.0) < 0.3)          # empirical matches analytic
    # halving-then-tenthing N multiplies analytic variance by ~10
    assert res.var_lambda_analytic[0] > res.var_lambda_analytic[-1]


def test_copies_to_certify_grows_toward_threshold(witness):
    n_easy = copies_to_certify(witness, partially_dephased_switch_process, 0.1)
    n_hard = copies_to_certify(witness, partially_dephased_switch_process, 0.6)
    assert 0 < n_easy < n_hard                         # closer to lambda* costs more copies


def test_krel_immune_raw_witness_inflates(witness):
    D = witness.shape[0]
    sig = dephased_switch_process() - ideal_quantum_switch_process()
    sighat = sig / np.linalg.norm(sig)
    N_conf = witness - float(np.sum(witness * sighat)) * sighat
    N_conf = N_conf - np.trace(N_conf) / D * np.eye(D)
    N_conf = N_conf / np.linalg.norm(N_conf)
    K_rel = admissible_direction(witness, [np.eye(D), N_conf], sig).K_rel
    fp = false_positive_under_drift(
        K_rel, witness, dephased_switch_process(), N_conf * float(np.linalg.norm(sig)),
        drift_grid=(0.0, 0.4, 0.8), N=5000, n_trials=500)
    # K_rel stays near alpha; raw witness inflates toward 1 under drift
    assert np.all(fp.fp_rate_krel < 0.1)
    assert fp.fp_rate_raw[-1] > 0.5
    assert np.max(np.abs(fp.krel_signal_shift - fp.krel_signal_shift[0])) < 1e-9
