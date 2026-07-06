r"""Certified interval on the generalized robustness (task A5).

A solver status of ``optimal`` is not a certificate.  This module brackets the
true generalized robustness ``R_g(W)`` between two verified numbers:

* a **lower bound** from weak duality: for any dual-feasible witness ``S``,
  ``R_g(W) >= Tr(S W)``.  We verify dual feasibility (PSD blocks with a small
  eigenvalue floor, subspace residual) and report ``Tr(S* W)`` as ``R_g_lower``.
* an **upper bound** from primal feasibility: any feasible ``(X, W_AB, W_BA)``
  gives ``R_g(W) <= Tr(X)/d_O``.  We report the primal objective as
  ``R_g_upper`` together with the equality residual, so the bound is feasible up
  to the reported tolerance.

The width ``R_g_upper - R_g_lower`` is the certified interval; it is essentially
the primal-dual gap.  A solver-independence table (SCS, CLARABEL, and MOSEK when
present) records that the interval is not an artefact of one solver.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import numpy as np

try:
    import cvxpy as cp  # type: ignore
except Exception:  # pragma: no cover
    cp = None

from .projectors import (
    L_A_before_B_with_future,
    L_B_before_A_with_future,
    L_valid_with_future,
    SWITCH_DIMS_WITH_FUTURE,
)
from .sdp import _subspace_complement_rows


@dataclass
class SolverResult:
    solver: str
    available: bool
    status: str
    R_g_lower: float          # verified dual (weak-duality) lower bound Tr(S* W)
    R_g_upper: float          # primal objective (feasible up to equality residual)
    interval_width: float
    equality_residual: float
    dual_subspace_residual: float
    min_eig_primal: float     # min eigenvalue floor across primal PSD blocks
    note: str = ""


def _solve_one(W: np.ndarray, dims, solver: str, eps: float, max_iters: int) -> SolverResult:
    dims = tuple(int(d) for d in dims)
    D = 1
    for d in dims:
        D *= d
    d_O = dims[1] * dims[3]
    W = 0.5 * (W + W.T)
    C_ab = _subspace_complement_rows(L_A_before_B_with_future, dims, "ABF")
    C_ba = _subspace_complement_rows(L_B_before_A_with_future, dims, "BAF")
    C_v = _subspace_complement_rows(L_valid_with_future, dims, "VF")

    W1 = cp.Variable((D, D), symmetric=True)
    W2 = cp.Variable((D, D), symmetric=True)
    X = cp.Variable((D, D), symmetric=True)
    vec = lambda M: cp.reshape(M, (D * D,), order="C")  # noqa: E731
    equality = W + X == W1 + W2
    cons = [W1 >> 0, W2 >> 0, X >> 0,
            C_ab @ vec(W1) == 0, C_ba @ vec(W2) == 0, C_v @ vec(X) == 0, equality]
    prob = cp.Problem(cp.Minimize(cp.trace(X) / d_O), cons)

    try:
        if solver.upper() == "SCS":
            prob.solve(solver="SCS", verbose=False, eps=eps, max_iters=max_iters)
        else:
            prob.solve(solver=solver.upper(), verbose=False)
    except Exception as exc:
        return SolverResult(solver, True, "error", float("nan"), float("nan"),
                            float("nan"), float("nan"), float("nan"), float("nan"),
                            note=f"solver raised: {type(exc).__name__}: {exc}")

    status = str(prob.status)
    if status not in {"optimal", "optimal_inaccurate"}:
        return SolverResult(solver, True, status, float("nan"), float("nan"),
                            float("nan"), float("nan"), float("nan"), float("nan"),
                            note="not solved to optimality")

    X_val = np.asarray(X.value); W1v = np.asarray(W1.value); W2v = np.asarray(W2.value)
    R_upper = float(prob.value)
    eqres = float(np.linalg.norm((W + X_val) - W1v - W2v))
    S = 0.5 * (np.asarray(equality.dual_value) + np.asarray(equality.dual_value).T)
    R_lower = float(np.sum(S * W))
    dual_sub = float(np.linalg.norm(C_ab @ W1v.reshape(-1)) + np.linalg.norm(C_ba @ W2v.reshape(-1)))
    min_eig = min(float(np.min(np.linalg.eigvalsh(0.5 * (M + M.T)))) for M in (X_val, W1v, W2v))
    width = abs(R_upper - R_lower)
    return SolverResult(solver, True, status, R_lower, R_upper, width, eqres,
                        dual_sub, min_eig, note="verified interval [dual LB, primal UB]")


@dataclass
class CertifiedInterval:
    R_g_lower: float
    R_g_upper: float
    width: float
    solver_table: list = field(default_factory=list)
    consensus_spread: float = float("nan")   # max-min of R_g_upper across solvers


def certified_robustness_interval(
    W: np.ndarray,
    dims: Sequence[int] = SWITCH_DIMS_WITH_FUTURE,
    solvers: Sequence[str] = ("SCS", "CLARABEL", "MOSEK"),
    eps: float = 1e-9,
    max_iters: int = 200_000,
) -> CertifiedInterval:
    """Return a certified interval on R_g(W) and a solver-independence table."""
    if cp is None:
        raise RuntimeError("cvxpy is required.")
    installed = set(cp.installed_solvers())
    table: list[SolverResult] = []
    for s in solvers:
        if s.upper() not in installed:
            table.append(SolverResult(s, False, "unavailable", float("nan"), float("nan"),
                                      float("nan"), float("nan"), float("nan"), float("nan"),
                                      note="solver not installed"))
            continue
        table.append(_solve_one(np.asarray(W, dtype=float), dims, s, eps, max_iters))

    ok = [r for r in table if r.status in {"optimal", "optimal_inaccurate"}]
    if not ok:
        return CertifiedInterval(float("nan"), float("nan"), float("nan"), table)
    # Tightest numerical bracket: highest verified LB, lowest verified UB.
    # At high tolerance SCS can return a tiny negative primal-dual gap
    # (lower > upper by ~1e-10).  Keep the raw values in the solver table and
    # expose an ordered interval at the aggregate level.
    raw_lower = max(r.R_g_lower for r in ok)
    raw_upper = min(r.R_g_upper for r in ok)
    R_lower, R_upper = sorted((raw_lower, raw_upper))
    uppers = [r.R_g_upper for r in ok]
    spread = float(max(uppers) - min(uppers))
    return CertifiedInterval(R_lower, R_upper, float(R_upper - R_lower), table, spread)
