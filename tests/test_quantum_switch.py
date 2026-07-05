"""Tests for the ideal quantum switch and the with-future projective structure.

Convention under test: Araújo, Branciard, Costa, Feix, Giarmatzi, Brukner,
NJP 17, 102001 (2015); systems [AI, AO, BI, BO, F], F = F_target ⊗ F_control.
The decisive external benchmark is the generalized robustness ≈ 0.5454.
"""
from __future__ import annotations

import numpy as np
import pytest

from deltawkrel.projectors import (
    ProcessDims,
    SWITCH_DIMS_WITH_FUTURE,
    L_A_before_B,
    L_A_before_B_with_future,
    L_B_before_A,
    L_B_before_A_with_future,
    L_valid_bipartite,
    L_valid_with_future,
    is_psd,
    random_hermitian,
    trace_replace,
    trace_replace_nd,
)
from deltawkrel.switch_models import (
    dephased_switch_process,
    ideal_quantum_switch_process,
    switch_branch_process,
)

D5 = SWITCH_DIMS_WITH_FUTURE
D = int(np.prod(D5))


# ------------------------------------------------------------------
# Fast n-partite trace-and-replace
# ------------------------------------------------------------------


def test_trace_replace_nd_agrees_with_audited_4party_map():
    dims4 = ProcessDims(2, 2, 2, 2)
    W = random_hermitian(dims4, seed=3).real
    for systems in ([0], [1, 2], [0, 3], [0, 1, 2, 3]):
        fast = trace_replace_nd(W, dims4.dims, systems)
        slow = trace_replace(W, dims4, systems)
        assert np.allclose(fast, slow, atol=1e-11), f"mismatch for systems={systems}"


def test_trace_replace_nd_validation_and_trivial_cases():
    with pytest.raises(ValueError):
        trace_replace_nd(np.eye(4), (2, 2, 2), [0])
    with pytest.raises(ValueError):
        trace_replace_nd(np.eye(8), (2, 2, 2), [5])
    with pytest.raises(ValueError):
        trace_replace_nd(np.eye(4), (2, 0, 2), [0])
    W = np.arange(16.0).reshape(4, 4)
    out = trace_replace_nd(W, (2, 2), [])
    assert np.allclose(out, W) and out is not W
    # Trivial dimension-1 system is a no-op.
    assert np.allclose(trace_replace_nd(W, (2, 2, 1), [2]), W)


# ------------------------------------------------------------------
# With-future projectors
# ------------------------------------------------------------------


def _random_symmetric_5(seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(D, D))
    return 0.5 * (X + X.T)


@pytest.mark.parametrize(
    "P", [L_A_before_B_with_future, L_B_before_A_with_future, L_valid_with_future]
)
def test_with_future_projectors_are_idempotent_and_self_adjoint(P):
    X = _random_symmetric_5(1)
    Y = _random_symmetric_5(2)
    PX = P(X, D5)
    assert np.allclose(P(PX, D5), PX, atol=1e-10)
    # Self-adjointness: <PX, Y> = <X, PY> (orthogonal projection).
    assert np.isclose(float(np.sum(PX * Y)), float(np.sum(X * P(Y, D5))), atol=1e-9)


def test_comb_subspaces_are_inside_valid_with_future():
    X = _random_symmetric_5(3)
    for P in (L_A_before_B_with_future, L_B_before_A_with_future):
        PX = P(X, D5)
        assert np.allclose(L_valid_with_future(PX, D5), PX, atol=1e-9)


def test_with_future_projectors_reduce_to_bipartite_for_trivial_future():
    """With d_F = 1 the with-future formulas must equal the bipartite projectors."""
    dims5 = (2, 2, 2, 2, 1)
    dims4 = ProcessDims(2, 2, 2, 2)
    W = random_hermitian(dims4, seed=4).real
    assert np.allclose(L_A_before_B_with_future(W, dims5), L_A_before_B(W, dims4), atol=1e-10)
    assert np.allclose(L_B_before_A_with_future(W, dims5), L_B_before_A(W, dims4), atol=1e-10)
    assert np.allclose(L_valid_with_future(W, dims5), L_valid_bipartite(W, dims4), atol=1e-10)


def test_with_future_projectors_reject_wrong_dims():
    with pytest.raises(ValueError):
        L_A_before_B_with_future(np.eye(16), (2, 2, 2, 2))


# ------------------------------------------------------------------
# Ideal quantum switch construction
# ------------------------------------------------------------------


def test_switch_is_rank_one_psd_with_correct_trace():
    W = ideal_quantum_switch_process()
    assert W.shape == (64, 64)
    assert np.allclose(W, W.T)
    assert np.isclose(np.trace(W), 4.0)  # Tr(W) = d_O = d_AO * d_BO
    eigenvalues = np.linalg.eigvalsh(W)
    assert eigenvalues[-1] == pytest.approx(4.0, abs=1e-10)  # pure |w><w|, <w|w>=4
    assert np.all(eigenvalues[:-1] < 1e-10)
    assert is_psd(W)


def test_switch_input_state_validation():
    with pytest.raises(ValueError):
        ideal_quantum_switch_process(np.zeros(2))
    with pytest.raises(ValueError):
        ideal_quantum_switch_process(np.array([1.0, 0.0, 0.0]))
    W = ideal_quantum_switch_process(np.array([3.0, 4.0]))  # normalized internally
    assert np.isclose(np.trace(W), 4.0)


def test_switch_is_valid_but_not_a_fixed_order_comb():
    W = ideal_quantum_switch_process()
    assert np.allclose(L_valid_with_future(W, D5), W, atol=1e-9)
    assert not np.allclose(L_A_before_B_with_future(W, D5), W, atol=1e-6)
    assert not np.allclose(L_B_before_A_with_future(W, D5), W, atol=1e-6)


def test_switch_branches_are_fixed_order_combs():
    Wab = switch_branch_process("AB")
    Wba = switch_branch_process("BA")
    for W, P in ((Wab, L_A_before_B_with_future), (Wba, L_B_before_A_with_future)):
        assert np.isclose(np.trace(W), 4.0)
        assert is_psd(W)
        assert np.allclose(P(W, D5), W, atol=1e-9)
    with pytest.raises(ValueError):
        switch_branch_process("XY")


def test_dephased_switch_is_the_branch_mixture():
    Wdeph = dephased_switch_process()
    expected = 0.5 * (switch_branch_process("AB") + switch_branch_process("BA"))
    assert np.allclose(Wdeph, expected)
    assert np.isclose(np.trace(Wdeph), 4.0)
    # Dephasing = the ideal switch with control coherences removed.
    W = ideal_quantum_switch_process().reshape(2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2).copy()
    for fc_row in range(2):
        for fc_col in range(2):
            if fc_row != fc_col:
                W[:, :, :, :, :, fc_row, :, :, :, :, :, fc_col] = 0.0
    assert np.allclose(W.reshape(64, 64), Wdeph, atol=1e-12)


def test_reduced_switch_branch_matches_bipartite_fixed_order_subspace():
    """Tracing out the future of a switch branch lands in the bipartite comb."""
    dims4 = ProcessDims(2, 2, 2, 2)
    Wab5 = switch_branch_process("AB")
    T = Wab5.reshape(16, 4, 16, 4)
    Wab4 = np.einsum("iaja->ij", T)  # partial trace over F
    assert np.isclose(np.trace(Wab4), 4.0)
    assert np.allclose(L_A_before_B(Wab4, dims4), Wab4, atol=1e-9)


# ------------------------------------------------------------------
# Published benchmark (requires cvxpy)
# ------------------------------------------------------------------


@pytest.fixture(scope="module")
def sdp_module():
    pytest.importorskip("cvxpy")
    from deltawkrel import sdp

    return sdp


def test_switch_generalized_robustness_reproduces_published_value(sdp_module):
    W = ideal_quantum_switch_process()
    diag = sdp_module.solve_switch_generalized_robustness(W)
    assert diag.status in {"optimal", "optimal_inaccurate"}
    # External benchmark: 0.5454 (Araújo et al., NJP 17, 102001 (2015)).
    assert diag.objective_value == pytest.approx(
        sdp_module.SWITCH_GENERALIZED_ROBUSTNESS_REFERENCE,
        abs=sdp_module.SWITCH_BENCHMARK_TOLERANCE,
    )
    # Regression pin on the computed value itself.
    assert diag.objective_value == pytest.approx(0.545351, abs=5e-4)
    # Dual certificate: Tr(S W) equals the primal optimum (complementarity).
    assert diag.witness_certificate_gap < 1e-6
    assert diag.equality_residual_fro < 1e-6
    assert diag.subspace_residual_fro < 1e-6
    assert diag.min_eig_W_AB > -1e-7 and diag.min_eig_W_BA > -1e-7
    assert diag.solver_version and diag.cvxpy_version


def test_dephased_switch_is_causally_separable(sdp_module):
    diag = sdp_module.solve_switch_generalized_robustness(dephased_switch_process())
    assert diag.status in {"optimal", "optimal_inaccurate"}
    assert abs(diag.objective_value) < 1e-6


def test_switch_random_robustness_regression_pin(sdp_module):
    diag = sdp_module.solve_switch_random_robustness(ideal_quantum_switch_process())
    assert diag.status in {"optimal", "optimal_inaccurate"}
    assert diag.objective_value == pytest.approx(1.9908, abs=5e-3)
    assert diag.t_white == pytest.approx(diag.objective_value, abs=1e-8)


def test_switch_solver_rejects_wrong_shape(sdp_module):
    with pytest.raises(ValueError):
        sdp_module.solve_switch_generalized_robustness(np.eye(16))
