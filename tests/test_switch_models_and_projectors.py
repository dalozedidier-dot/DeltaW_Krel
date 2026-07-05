import numpy as np

from deltawkrel.projectors import (
    ProcessDims,
    L_A_before_B,
    L_B_before_A,
    L_valid_bipartite,
    assert_trace_convention,
    is_psd,
)
from deltawkrel.switch_models import (
    fixed_order_A_before_B_process,
    fixed_order_B_before_A_process,
    causally_separable_mixture,
    ideal_quantum_switch_process,
)


def test_fixed_order_processes_trace_psd_and_membership():
    dims = ProcessDims(2, 2, 2, 2)
    Wab = fixed_order_A_before_B_process(dims)
    Wba = fixed_order_B_before_A_process(dims)
    for W in (Wab, Wba):
        assert_trace_convention(W, dims)
        assert is_psd(W)
        assert np.allclose(L_valid_bipartite(W, dims), W, atol=1e-8)
    assert np.allclose(L_A_before_B(Wab, dims), Wab, atol=1e-8)
    assert np.allclose(L_B_before_A(Wba, dims), Wba, atol=1e-8)


def test_causally_separable_mixture_is_valid_and_psd():
    dims = ProcessDims(2, 2, 2, 2)
    W = causally_separable_mixture(dims, q=0.25)
    assert_trace_convention(W, dims)
    assert is_psd(W)
    assert np.allclose(L_valid_bipartite(W, dims), W, atol=1e-8)


def test_ideal_quantum_switch_is_a_real_process_matrix():
    """The switch is implemented for real: pure, normalized, PSD at D=64."""
    W = ideal_quantum_switch_process()
    assert W.shape == (64, 64)
    assert np.isclose(np.trace(W), 4.0)
    assert is_psd(W)
    eigenvalues = np.linalg.eigvalsh(W)
    assert eigenvalues[-1] > 3.99  # rank-one with <w|w> = 4
