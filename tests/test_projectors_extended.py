"""Extended algebraic tests for the bipartite process-matrix projectors.

These tests audit the trace-and-replace map and the derived projectors as
linear-algebra objects: linearity, idempotence, self-adjointness (orthogonal
projection), fixed points, subspace inclusions and input validation.
"""
from __future__ import annotations

import numpy as np
import pytest

from deltawkrel.projectors import (
    AI,
    AO,
    BI,
    BO,
    ProcessDims,
    _as_dims,
    assert_trace_convention,
    d_output,
    hilbert_dim,
    is_idempotent,
    is_psd,
    L_A_before_B,
    L_B_before_A,
    L_valid_bipartite,
    maximally_entangled_projector,
    random_hermitian,
    superoperator_matrix,
    symmetrize,
    trace_replace,
    trrep,
    white_noise_process,
)

DIMS = ProcessDims(2, 2, 2, 2)


# ------------------------------------------------------------------
# ProcessDims and dimension helpers
# ------------------------------------------------------------------


def test_process_dims_properties():
    dims = ProcessDims(2, 3, 4, 5)
    assert dims.dims == (2, 3, 4, 5)
    assert dims.hilbert_dim == 2 * 3 * 4 * 5
    assert dims.d_O == 3 * 5


def test_dimension_helpers_accept_sequences():
    assert hilbert_dim([2, 2, 2, 2]) == 16
    assert d_output([2, 3, 2, 5]) == 15
    assert d_output(DIMS) == 4


def test_as_dims_rejects_bad_input():
    with pytest.raises(ValueError):
        _as_dims([2, 2, 2])
    with pytest.raises(ValueError):
        _as_dims([2, 2, 2, 0])
    with pytest.raises(ValueError):
        _as_dims([2, 2, 2, -1])


# ------------------------------------------------------------------
# trace_replace: the core map
# ------------------------------------------------------------------


def test_trace_replace_rejects_wrong_shape_and_bad_system():
    with pytest.raises(ValueError):
        trace_replace(np.eye(4), DIMS, [AI])
    with pytest.raises(ValueError):
        trace_replace(np.eye(16), DIMS, [7])
    with pytest.raises(ValueError):
        trace_replace(np.eye(16), DIMS, [-1])


def test_trace_replace_empty_systems_is_identity_copy():
    W = random_hermitian(DIMS, seed=5)
    out = trace_replace(W, DIMS, [])
    assert np.allclose(out, W)
    assert out is not W


def test_trace_replace_preserves_trace():
    W = random_hermitian(DIMS, seed=6)
    for systems in ([AI], [AO], [AI, BO], [AI, AO, BI, BO]):
        out = trace_replace(W, DIMS, systems)
        assert np.isclose(np.trace(out), np.trace(W))


def test_trace_replace_is_idempotent_and_linear():
    W1 = random_hermitian(DIMS, seed=7)
    W2 = random_hermitian(DIMS, seed=8)
    systems = [AO, BI]
    once = trace_replace(W1, DIMS, systems)
    assert np.allclose(trace_replace(once, DIMS, systems), once, atol=1e-10)
    lhs = trace_replace(2.0 * W1 - 3.0 * W2, DIMS, systems)
    rhs = 2.0 * trace_replace(W1, DIMS, systems) - 3.0 * trace_replace(W2, DIMS, systems)
    assert np.allclose(lhs, rhs, atol=1e-10)


def test_trace_replace_on_kron_product_replaces_first_factor():
    """On W = A ⊗ B with subsystem grouping (AI) vs (AO,BI,BO), replacing AI
    must give (I/d) Tr(A) ⊗ B."""
    rng = np.random.default_rng(9)
    A = symmetrize(rng.normal(size=(2, 2)))
    B = symmetrize(rng.normal(size=(8, 8)))
    W = np.kron(A, B)
    out = trace_replace(W, DIMS, [AI])
    expected = np.kron(np.eye(2) * (np.trace(A) / 2.0), B)
    assert np.allclose(out, expected, atol=1e-10)


def test_trace_replace_full_replacement_gives_scaled_identity():
    W = random_hermitian(DIMS, seed=10)
    out = trace_replace(W, DIMS, [AI, AO, BI, BO])
    expected = np.eye(16) * (np.trace(W) / 16.0)
    assert np.allclose(out, expected, atol=1e-10)


def test_trace_replace_supports_complex_and_nonuniform_dims():
    dims = ProcessDims(2, 3, 2, 3)
    W = random_hermitian(dims, seed=11)
    assert np.iscomplexobj(W)
    out = trace_replace(W, dims, [AO])
    assert out.shape == (dims.hilbert_dim, dims.hilbert_dim)
    assert np.allclose(out, out.T.conj(), atol=1e-10)
    assert np.allclose(trace_replace(out, dims, [AO]), out, atol=1e-10)


def test_trrep_is_alias():
    W = random_hermitian(DIMS, seed=12)
    assert np.allclose(trrep(W, DIMS, [BI]), trace_replace(W, DIMS, [BI]))


# ------------------------------------------------------------------
# Projectors as superoperators
# ------------------------------------------------------------------


@pytest.mark.parametrize("projector", [L_valid_bipartite, L_A_before_B, L_B_before_A])
def test_projector_superoperators_are_orthogonal_projections(projector):
    P = superoperator_matrix(projector, DIMS, order="C")
    assert np.allclose(P @ P, P, atol=1e-8)
    assert np.allclose(P, P.T, atol=1e-8)
    eigenvalues = np.linalg.eigvalsh(0.5 * (P + P.T))
    assert np.all((np.abs(eigenvalues) < 1e-7) | (np.abs(eigenvalues - 1.0) < 1e-7))


def test_fixed_order_subspaces_are_inside_valid_subspace():
    W = random_hermitian(DIMS, seed=13).real
    for P in (L_A_before_B, L_B_before_A):
        PW = P(W, DIMS)
        assert np.allclose(L_valid_bipartite(PW, DIMS), PW, atol=1e-8)


def test_white_noise_is_fixed_point_of_all_projectors():
    W = white_noise_process(DIMS)
    for P in (L_valid_bipartite, L_A_before_B, L_B_before_A):
        assert np.allclose(P(W, DIMS), W, atol=1e-10)


def test_superoperator_orders_agree_on_symmetric_input():
    W = random_hermitian(DIMS, seed=14).real
    P_C = superoperator_matrix(L_valid_bipartite, DIMS, order="C")
    P_F = superoperator_matrix(L_valid_bipartite, DIMS, order="F")
    out_C = (P_C @ W.reshape(-1, order="C")).reshape(16, 16, order="C")
    out_F = (P_F @ W.reshape(-1, order="F")).reshape(16, 16, order="F")
    assert np.allclose(out_C, out_F, atol=1e-8)
    assert np.allclose(out_C, L_valid_bipartite(W, DIMS), atol=1e-8)


def test_superoperator_matrix_rejects_bad_order():
    with pytest.raises(ValueError):
        superoperator_matrix(L_valid_bipartite, DIMS, order="X")


def test_is_idempotent_helper():
    assert is_idempotent(L_valid_bipartite, DIMS, seed=3)
    assert is_idempotent(L_A_before_B, DIMS, seed=4)
    assert is_idempotent(L_B_before_A, DIMS, seed=5)


def test_projector_dimension_counts_match_manuscript():
    """dim(L_AB) + dim(L_BA) - dim(L_AB ∩ L_BA) <= dim(L_valid)."""
    P_valid = superoperator_matrix(L_valid_bipartite, DIMS)
    P_AB = superoperator_matrix(L_A_before_B, DIMS)
    P_BA = superoperator_matrix(L_B_before_A, DIMS)
    rank = lambda P: int(round(float(np.trace(P))))  # noqa: E731
    assert 0 < rank(P_AB) <= rank(P_valid)
    assert 0 < rank(P_BA) <= rank(P_valid)
    # A≺B and B≺A subspaces are exchanged by the A<->B swap: equal dimensions.
    assert rank(P_AB) == rank(P_BA)


# ------------------------------------------------------------------
# Auxiliary constructors and checks
# ------------------------------------------------------------------


def test_white_noise_process_properties():
    W = white_noise_process(DIMS)
    assert np.isclose(np.trace(W), d_output(DIMS))
    assert is_psd(W)
    assert_trace_convention(W, DIMS)


def test_maximally_entangled_projector_properties():
    P = maximally_entangled_projector(2)
    assert P.shape == (4, 4)
    assert np.isclose(np.trace(P), 2.0)
    assert is_psd(P)
    assert np.allclose(P @ P, 2.0 * P, atol=1e-12)  # rank-one, unnormalized
    with pytest.raises(ValueError):
        maximally_entangled_projector(2, 3)


def test_random_hermitian_is_hermitian():
    W = random_hermitian(DIMS, seed=21)
    assert np.allclose(W, W.T.conj())
    W2 = random_hermitian(DIMS, seed=21)
    assert np.allclose(W, W2)  # deterministic given seed


def test_is_psd_detects_negative_eigenvalues():
    assert is_psd(np.eye(3))
    assert not is_psd(-np.eye(3))
    assert is_psd(np.zeros((3, 3)))


def test_assert_trace_convention_raises_on_violation():
    with pytest.raises(AssertionError):
        assert_trace_convention(np.eye(16), DIMS)  # Tr=16 != d_O=4


def test_symmetrize_hermitian_part():
    X = np.array([[1.0, 2.0 + 1.0j], [0.0, 3.0]])
    S = symmetrize(X)
    assert np.allclose(S, S.T.conj())
