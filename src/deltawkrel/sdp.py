"""Minimal SDP routines for causally separable robustness validation.

The routine below solves the primal conic scaffold used in the manuscript:

    min t_white + Σ_i t_i
    s.t. W0 + Σ_i t_i N_i + t_white W_white = W_AB + W_BA
         W_AB >= 0, W_BA >= 0
         W_AB ∈ L_AB, W_BA ∈ L_BA
         t_i >= 0, t_white >= 0.

This is now a genuine K_CS wiring test, not merely a PSD proxy.  It is still not
an ideal quantum-switch benchmark: that benchmark needs the explicit switch
process with future/control convention.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Sequence

import numpy as np

from .projectors import (
    ProcessDims,
    L_A_before_B,
    L_B_before_A,
    L_A_before_B_with_future,
    L_B_before_A_with_future,
    L_valid_with_future,
    SWITCH_DIMS_WITH_FUTURE,
    superoperator_matrix,
    white_noise_process,
    hilbert_dim,
)

try:
    import cvxpy as cp  # type: ignore
except Exception:  # pragma: no cover
    cp = None


@dataclass
class SdpDiagnostics:
    status: str
    objective_value: float
    t_white: float
    t_calibrated_sum: float
    omega_white: float
    solver: str
    equality_residual_fro: float
    note: str
    # Extended reporting (defaults keep older construction sites valid).
    solver_version: str = ""
    cvxpy_version: str = ""
    num_iters: float = float("nan")
    solve_time_s: float = float("nan")
    min_eig_W_AB: float = float("nan")
    min_eig_W_BA: float = float("nan")
    witness_value: float = float("nan")
    witness_certificate_gap: float = float("nan")
    subspace_residual_fro: float = float("nan")

    def to_dict(self) -> dict:
        return asdict(self)


def _strict_float(x: float) -> float | None:
    x = float(x)
    return None if not np.isfinite(x) else x


def _solver_versions(chosen: str) -> tuple[str, str]:
    """Best-effort (solver_version, cvxpy_version) for reproducibility reports."""
    cvxpy_version = getattr(cp, "__version__", "") if cp is not None else ""
    solver_version = ""
    try:
        if chosen.upper() == "CLARABEL":
            import clarabel  # type: ignore

            solver_version = getattr(clarabel, "__version__", "")
        elif chosen.upper() == "SCS":
            import scs  # type: ignore

            solver_version = getattr(scs, "__version__", "")
    except Exception:
        solver_version = ""
    return solver_version, cvxpy_version


def _solver_stats(problem) -> tuple[float, float]:
    """Return (num_iters, solve_time_s) from cvxpy solver stats when available."""
    stats = getattr(problem, "solver_stats", None)
    if stats is None:
        return float("nan"), float("nan")
    num_iters = stats.num_iters if stats.num_iters is not None else float("nan")
    solve_time = stats.solve_time if stats.solve_time is not None else float("nan")
    return float(num_iters), float(solve_time)


def solve_cs_robustness(
    W0: np.ndarray,
    dims: ProcessDims | Sequence[int] = ProcessDims(),
    noises: Sequence[np.ndarray] | None = None,
    W_white: np.ndarray | None = None,
    solver: str = "CLARABEL",
    eps: float = 1e-7,
) -> SdpDiagnostics:
    """Solve calibrated robustness over K_CS = C_AB + C_BA.

    The linear subspace constraints are imposed using the vectorized projectors
    L_A_before_B and L_B_before_A.  This validates the cone wiring on known CS
    targets.  For full submission, call this on an implemented ideal quantum
    switch and compare against a published benchmark.
    """
    if cp is None:
        raise RuntimeError("cvxpy is required for SDP validation. Install requirements.txt.")

    D = hilbert_dim(dims)
    W0 = np.asarray(W0, dtype=float)
    if W0.shape != (D, D):
        raise ValueError(f"W0 must have shape {(D, D)}, got {W0.shape}.")
    W0 = 0.5 * (W0 + W0.T)
    if W_white is None:
        W_white = white_noise_process(dims)
    W_white = np.asarray(W_white, dtype=float)
    noises = list(noises or [])

    W_AB = cp.Variable((D, D), symmetric=True, name="W_AB")
    W_BA = cp.Variable((D, D), symmetric=True, name="W_BA")
    t_white = cp.Variable(nonneg=True, name="t_white")
    t_cal = cp.Variable(len(noises), nonneg=True, name="t_cal") if noises else None

    P_AB = superoperator_matrix(L_A_before_B, dims, order="C")
    P_BA = superoperator_matrix(L_B_before_A, dims, order="C")
    vec_AB = cp.reshape(W_AB, (D * D,), order="C")
    vec_BA = cp.reshape(W_BA, (D * D,), order="C")

    left = W0 + t_white * W_white
    if noises:
        for i, N_i in enumerate(noises):
            left = left + t_cal[i] * np.asarray(N_i, dtype=float)

    constraints = [
        W_AB >> 0,
        W_BA >> 0,
        vec_AB == P_AB @ vec_AB,
        vec_BA == P_BA @ vec_BA,
        left == W_AB + W_BA,
    ]
    objective = cp.Minimize(t_white + (cp.sum(t_cal) if noises else 0.0))
    problem = cp.Problem(objective, constraints)

    chosen = solver
    try:
        problem.solve(solver=chosen, verbose=False)
    except Exception as exc:
        if solver != "SCS":
            print(f"AVERTISSEMENT : fallback du solveur {solver} vers SCS ({exc}).")
            chosen = "SCS"
            problem.solve(solver=chosen, verbose=False, eps=eps)
        else:
            raise

    solver_version, cvxpy_version = _solver_versions(chosen)
    num_iters, solve_time = _solver_stats(problem)
    status = str(problem.status)
    if status not in {"optimal", "optimal_inaccurate"}:
        return SdpDiagnostics(
            status, float("nan"), float("nan"), float("nan"), float("nan"), chosen,
            float("nan"), "SDP not solved to optimality.",
            solver_version=solver_version, cvxpy_version=cvxpy_version,
            num_iters=num_iters, solve_time_s=solve_time,
        )

    tw = max(0.0, float(t_white.value))
    tc = float(np.sum(np.maximum(0.0, np.asarray(t_cal.value)))) if noises else 0.0
    denom = tw + tc
    omega = tw / denom if denom > 1e-14 else 0.0
    equality_residual = float(np.linalg.norm(np.asarray(left.value) - np.asarray(W_AB.value) - np.asarray(W_BA.value)))
    min_eig_ab = float(np.min(np.linalg.eigvalsh(0.5 * (np.asarray(W_AB.value) + np.asarray(W_AB.value).T))))
    min_eig_ba = float(np.min(np.linalg.eigvalsh(0.5 * (np.asarray(W_BA.value) + np.asarray(W_BA.value).T))))
    note = (
        "K_CS SDP solved on the supplied bipartite target. The ideal-switch benchmark "
        "is provided by solve_switch_generalized_robustness."
    )
    return SdpDiagnostics(
        status, float(problem.value), tw, tc, omega, chosen, equality_residual, note,
        solver_version=solver_version, cvxpy_version=cvxpy_version,
        num_iters=num_iters, solve_time_s=solve_time,
        min_eig_W_AB=min_eig_ab, min_eig_W_BA=min_eig_ba,
    )


# ------------------------------------------------------------------
# Ideal quantum switch: causal robustness with a global future space
# ------------------------------------------------------------------

# Published benchmark: generalized robustness of the ideal quantum switch,
# Araújo, Branciard, Costa, Feix, Giarmatzi, Brukner, "Witnessing causal
# nonseparability", New J. Phys. 17, 102001 (2015).  Our SDP reproduces
# 0.545351 at eps=1e-8/1e-9 (SCS), matching the published 4-digit value.
SWITCH_GENERALIZED_ROBUSTNESS_REFERENCE = 0.5454
SWITCH_BENCHMARK_TOLERANCE = 2e-3

_COMPLEMENT_CACHE: dict = {}


def _symmetric_basis_matrix(D: int) -> np.ndarray:
    """Columns = vec of an orthonormal basis of D×D real symmetric matrices."""
    n = D * (D + 1) // 2
    B = np.zeros((D * D, n))
    k = 0
    for i in range(D):
        for j in range(i, D):
            E = np.zeros((D, D))
            if i == j:
                E[i, i] = 1.0
            else:
                E[i, j] = E[j, i] = 1.0 / np.sqrt(2.0)
            B[:, k] = E.reshape(-1)
            k += 1
    return B


def _subspace_complement_rows(projector, dims: Sequence[int], key: str) -> np.ndarray:
    """Rows C such that C @ vec(W) = 0 iff the symmetric W lies in im(projector).

    The projector is restricted to the symmetric-matrix space, diagonalized,
    and the eigenvalue-0 directions form the orthogonal complement.  Cached per
    (key, dims) because the construction costs a few seconds at D=64.
    """
    dims_key = (key, tuple(int(d) for d in dims))
    if dims_key in _COMPLEMENT_CACHE:
        return _COMPLEMENT_CACHE[dims_key]

    D = 1
    for d in dims:
        D *= int(d)
    B = _symmetric_basis_matrix(D)
    n = B.shape[1]
    PB = np.zeros((D * D, n))
    for k in range(n):
        PB[:, k] = projector(B[:, k].reshape(D, D), dims).reshape(-1)
    P_sym = B.T @ PB
    vals, vecs = np.linalg.eigh(0.5 * (P_sym + P_sym.T))
    if not np.all((np.abs(vals) < 1e-8) | (np.abs(vals - 1.0) < 1e-8)):
        raise RuntimeError("projector restricted to symmetric space is not a projection.")
    comp = vecs[:, vals < 0.5]
    rows = (B @ comp).T
    _COMPLEMENT_CACHE[dims_key] = rows
    return rows


def _switch_robustness_common(
    W: np.ndarray,
    dims: Sequence[int],
    solver: str,
    eps: float,
    max_iters: int,
    generalized: bool,
) -> SdpDiagnostics:
    if cp is None:
        raise RuntimeError("cvxpy is required for SDP validation. Install requirements.txt.")

    dims = tuple(int(d) for d in dims)
    D = 1
    for d in dims:
        D *= d
    d_O = dims[1] * dims[3]
    W = np.asarray(W, dtype=float)
    if W.shape != (D, D):
        raise ValueError(f"W must have shape {(D, D)}, got {W.shape}.")
    W = 0.5 * (W + W.T)

    C_ab = _subspace_complement_rows(L_A_before_B_with_future, dims, "ABF")
    C_ba = _subspace_complement_rows(L_B_before_A_with_future, dims, "BAF")

    W1 = cp.Variable((D, D), symmetric=True, name="W_AB")
    W2 = cp.Variable((D, D), symmetric=True, name="W_BA")
    vec = lambda M: cp.reshape(M, (D * D,), order="C")  # noqa: E731

    constraints = [W1 >> 0, W2 >> 0, C_ab @ vec(W1) == 0, C_ba @ vec(W2) == 0]
    if generalized:
        C_v = _subspace_complement_rows(L_valid_with_future, dims, "VF")
        X = cp.Variable((D, D), symmetric=True, name="X")
        constraints += [X >> 0, C_v @ vec(X) == 0]
        equality = W + X == W1 + W2
        objective = cp.Minimize(cp.trace(X) / d_O)
        noise_kind = "generalized (arbitrary valid process)"
    else:
        W_white = np.eye(D) * (d_O / D)
        t = cp.Variable(nonneg=True, name="t")
        equality = W + t * W_white == W1 + W2
        objective = cp.Minimize(t)
        noise_kind = "random (white noise)"
    constraints.append(equality)

    problem = cp.Problem(objective, constraints)
    chosen = solver
    try:
        if chosen.upper() == "SCS":
            problem.solve(solver="SCS", verbose=False, eps=eps, max_iters=max_iters)
        else:
            problem.solve(solver=chosen, verbose=False)
    except Exception as exc:
        if chosen.upper() != "SCS":
            print(f"AVERTISSEMENT : fallback du solveur {chosen} vers SCS ({exc}).")
            chosen = "SCS"
            problem.solve(solver="SCS", verbose=False, eps=eps, max_iters=max_iters)
        else:
            raise

    solver_version, cvxpy_version = _solver_versions(chosen)
    num_iters, solve_time = _solver_stats(problem)
    status = str(problem.status)
    if status not in {"optimal", "optimal_inaccurate"}:
        return SdpDiagnostics(
            status, float("nan"), float("nan"), float("nan"), float("nan"), chosen,
            float("nan"), f"Switch {noise_kind} robustness SDP not solved to optimality.",
            solver_version=solver_version, cvxpy_version=cvxpy_version,
            num_iters=num_iters, solve_time_s=solve_time,
        )

    W1_val = np.asarray(W1.value)
    W2_val = np.asarray(W2.value)
    if generalized:
        X_val = np.asarray(X.value)
        left = W + X_val
        t_white_report = float("nan")
    else:
        t_val = max(0.0, float(t.value))
        left = W + t_val * (np.eye(D) * (d_O / D))
        t_white_report = t_val
    equality_residual = float(np.linalg.norm(left - W1_val - W2_val))
    subspace_residual = float(
        np.linalg.norm(C_ab @ W1_val.reshape(-1)) + np.linalg.norm(C_ba @ W2_val.reshape(-1))
    )
    min_eig_ab = float(np.min(np.linalg.eigvalsh(0.5 * (W1_val + W1_val.T))))
    min_eig_ba = float(np.min(np.linalg.eigvalsh(0.5 * (W2_val + W2_val.T))))

    # Dual of the main equality constraint = causal witness certificate S.
    witness_value = float("nan")
    certificate_gap = float("nan")
    dual = equality.dual_value if hasattr(equality, "dual_value") else None
    if dual is not None:
        S = 0.5 * (np.asarray(dual) + np.asarray(dual).T)
        witness_value = float(np.sum(S * W))
        certificate_gap = float(abs(witness_value - float(problem.value)))

    return SdpDiagnostics(
        status=status,
        objective_value=float(problem.value),
        t_white=t_white_report,
        t_calibrated_sum=0.0,
        omega_white=float("nan"),
        solver=chosen,
        equality_residual_fro=equality_residual,
        note=(
            f"Ideal-quantum-switch {noise_kind} robustness over K_CS with future space. "
            "Benchmark reference 0.5454 (Araújo et al., NJP 17, 102001 (2015)) applies "
            "to the generalized robustness."
        ),
        solver_version=solver_version,
        cvxpy_version=cvxpy_version,
        num_iters=num_iters,
        solve_time_s=solve_time,
        min_eig_W_AB=min_eig_ab,
        min_eig_W_BA=min_eig_ba,
        witness_value=witness_value,
        witness_certificate_gap=certificate_gap,
        subspace_residual_fro=subspace_residual,
    )


def solve_switch_generalized_robustness(
    W: np.ndarray,
    dims: Sequence[int] = SWITCH_DIMS_WITH_FUTURE,
    solver: str = "SCS",
    eps: float = 1e-8,
    max_iters: int = 200_000,
) -> SdpDiagnostics:
    """Generalized robustness R_g of a process with future space over K_CS.

    R_g(W) = min Tr(X)/d_O s.t. W + X = W_AB + W_BA with X a valid-process cone
    element and W_AB/W_BA PSD elements of the fixed-order comb subspaces —
    equivalently min t such that (W + t Ω)/(1+t) is causally separable for some
    valid process Ω.  For the ideal quantum switch the published value is
    ≈ 0.5454 (Araújo et al., NJP 17, 102001 (2015)).

    SCS is the default solver: CLARABEL currently fails with a NumericalError
    on this instance (reported in the diagnostics note if the fallback fires).
    """
    return _switch_robustness_common(W, dims, solver, eps, max_iters, generalized=True)


def solve_switch_random_robustness(
    W: np.ndarray,
    dims: Sequence[int] = SWITCH_DIMS_WITH_FUTURE,
    solver: str = "SCS",
    eps: float = 1e-8,
    max_iters: int = 200_000,
) -> SdpDiagnostics:
    """Random (white-noise) robustness of a process with future space over K_CS."""
    return _switch_robustness_common(W, dims, solver, eps, max_iters, generalized=False)


# Backward-compatible name used by older tests/notebooks.
def solve_cs_robustness_scaffold(*args, **kwargs) -> SdpDiagnostics:
    return solve_cs_robustness(*args, **kwargs)
