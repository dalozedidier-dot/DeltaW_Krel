"""Extended tests for the fixed-order and mixture validation targets."""
from __future__ import annotations

import numpy as np
import pytest

from deltawkrel.projectors import (
    ProcessDims,
    L_A_before_B,
    L_B_before_A,
    L_valid_bipartite,
    assert_trace_convention,
    d_output,
    is_psd,
    white_noise_process,
)
from deltawkrel.switch_models import (
    _kron_all,
    causally_separable_mixture,
    fixed_order_A_before_B_process,
    fixed_order_B_before_A_process,
    ideal_quantum_switch_process,
    white_noise_validation_process,
)

DIMS = ProcessDims(2, 2, 2, 2)


def test_kron_all_matches_numpy():
    rng = np.random.default_rng(0)
    A, B, C = (rng.normal(size=(2, 2)) for _ in range(3))
    assert np.allclose(_kron_all([A, B, C]), np.kron(np.kron(A, B), C))
    assert np.allclose(_kron_all([]), np.array([[1.0]]))


def test_fixed_order_processes_are_distinct_and_ordered():
    Wab = fixed_order_A_before_B_process(DIMS)
    Wba = fixed_order_B_before_A_process(DIMS)
    assert not np.allclose(Wab, Wba)
    # A≺B process must NOT live in the B≺A subspace (and vice versa).
    assert not np.allclose(L_B_before_A(Wab, DIMS), Wab, atol=1e-8)
    assert not np.allclose(L_A_before_B(Wba, DIMS), Wba, atol=1e-8)


def test_fixed_order_processes_related_by_party_swap():
    """W_BA is the party-swap image of W_AB: same spectrum, same trace."""
    Wab = fixed_order_A_before_B_process(DIMS)
    Wba = fixed_order_B_before_A_process(DIMS)
    assert np.isclose(np.trace(Wab), np.trace(Wba))
    ev_ab = np.sort(np.linalg.eigvalsh(Wab))
    ev_ba = np.sort(np.linalg.eigvalsh(Wba))
    assert np.allclose(ev_ab, ev_ba, atol=1e-10)


def test_fixed_order_dimension_mismatch_raises():
    with pytest.raises(ValueError):
        fixed_order_A_before_B_process(ProcessDims(2, 2, 3, 2))  # AO != BI
    with pytest.raises(ValueError):
        fixed_order_B_before_A_process(ProcessDims(3, 2, 2, 2))  # BO != AI


def test_mixture_endpoints_and_convexity():
    Wab = fixed_order_A_before_B_process(DIMS)
    Wba = fixed_order_B_before_A_process(DIMS)
    assert np.allclose(causally_separable_mixture(DIMS, q=1.0), Wab)
    assert np.allclose(causally_separable_mixture(DIMS, q=0.0), Wba)
    W_half = causally_separable_mixture(DIMS, q=0.5)
    assert np.allclose(W_half, 0.5 * Wab + 0.5 * Wba)
    assert_trace_convention(W_half, DIMS)
    assert is_psd(W_half)
    assert np.allclose(L_valid_bipartite(W_half, DIMS), W_half, atol=1e-8)


def test_mixture_rejects_q_outside_unit_interval():
    with pytest.raises(ValueError):
        causally_separable_mixture(DIMS, q=-0.01)
    with pytest.raises(ValueError):
        causally_separable_mixture(DIMS, q=1.01)


def test_ideal_quantum_switch_lives_outside_bipartite_scaffold():
    """The full switch is a 64×64 with-future process, distinct per target state."""
    W0 = ideal_quantum_switch_process()
    W1 = ideal_quantum_switch_process(np.array([1.0, 1.0]) / np.sqrt(2.0))
    assert W0.shape == W1.shape == (64, 64)
    assert not np.allclose(W0, W1)
    for W in (W0, W1):
        assert np.isclose(np.trace(W), 4.0)
        assert is_psd(W)


def test_white_noise_validation_process_alias():
    assert np.allclose(white_noise_validation_process(DIMS), white_noise_process(DIMS))


def test_fixed_order_larger_dimensions():
    dims = ProcessDims(3, 3, 3, 3)
    W = fixed_order_A_before_B_process(dims)
    assert W.shape == (81, 81)
    assert np.isclose(np.trace(W), d_output(dims))
    assert is_psd(W)
    assert np.allclose(L_A_before_B(W, dims), W, atol=1e-8)
