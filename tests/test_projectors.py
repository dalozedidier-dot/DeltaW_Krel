import numpy as np

from deltawkrel.projectors import (
    ProcessDims,
    L_valid_bipartite,
    L_A_before_B,
    L_B_before_A,
    d_output,
    random_hermitian,
    superoperator_matrix,
    white_noise_process,
)


def test_projectors_are_idempotent():
    dims = ProcessDims(2, 2, 2, 2)
    W = random_hermitian(dims, seed=123).real
    for P in (L_valid_bipartite, L_A_before_B, L_B_before_A):
        PW = P(W, dims)
        assert np.allclose(P(PW, dims), PW, atol=1e-8)


def test_white_noise_trace_convention():
    dims = ProcessDims(2, 2, 2, 2)
    W = white_noise_process(dims)
    assert np.allclose(np.trace(W), d_output(dims), atol=1e-8)


def test_superoperator_idempotent_for_valid_projector():
    dims = ProcessDims(2, 2, 2, 2)
    P = superoperator_matrix(L_valid_bipartite, dims)
    assert np.allclose(P @ P, P, atol=1e-8)
