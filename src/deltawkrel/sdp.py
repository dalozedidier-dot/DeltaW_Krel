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

    def to_dict(self) -> dict:
        return asdict(self)


def _strict_float(x: float) -> float | None:
    x = float(x)
    return None if not np.isfinite(x) else x


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

    status = str(problem.status)
    if status not in {"optimal", "optimal_inaccurate"}:
        return SdpDiagnostics(status, float("nan"), float("nan"), float("nan"), float("nan"), chosen, float("nan"), "SDP not solved to optimality.")

    tw = max(0.0, float(t_white.value))
    tc = float(np.sum(np.maximum(0.0, np.asarray(t_cal.value)))) if noises else 0.0
    denom = tw + tc
    omega = tw / denom if denom > 1e-14 else 0.0
    equality_residual = float(np.linalg.norm(np.asarray(left.value) - np.asarray(W_AB.value) - np.asarray(W_BA.value)))
    note = "K_CS SDP solved on the supplied target. Ideal-switch published benchmark still required for submission."
    return SdpDiagnostics(status, float(problem.value), tw, tc, omega, chosen, equality_residual, note)


# Backward-compatible name used by older tests/notebooks.
def solve_cs_robustness_scaffold(*args, **kwargs) -> SdpDiagnostics:
    return solve_cs_robustness(*args, **kwargs)
