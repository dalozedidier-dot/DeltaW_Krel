r"""Certified single-direction causal witness for the ideal quantum switch.

This module fuses the two layers of the DeltaW/K_rel program on the *physical*
switch process, rather than on an abstract toy space:

* the **SDP layer**, which reproduces the published generalized robustness
  ``R_g(W_switch) approx 0.5454`` (Araujo et al., NJP 17, 102001 (2015)); and
* the **decision layer**, which preregisters a single relational direction and
  turns the falsification test into one scalar statistic.

Central result (a consequence of conic duality, verified numerically to machine
precision on the switch):

    Let ``S*`` be the dual-optimal causal witness of the generalized-robustness
    SDP at the reference process ``W_switch``.  For any affine process family
    ``W(lambda)`` the map ``lambda -> Tr(S* W(lambda))`` is *exactly affine*,
    equals ``R_g`` at the reference, and is a *certified lower bound* on the full
    SDP robustness ``R_g(W(lambda))`` everywhere (it is the supporting affine
    functional at the reference).  Its zero crossing ``lambda*`` is a certified
    causal-nonseparability threshold: for ``lambda < lambda*`` the process is
    provably causally nonseparable, using a single scalar instead of a fresh SDP
    or a full ``D^2`` process tomography at each point.

The admissible / relational direction ``K_rel`` is the component of ``S*``
orthogonal (in Hilbert-Schmidt inner product) to a preregistered set of
calibrated-noise directions.  It is first-order insensitive to those nuisance
directions, at the cost of a strictly weaker (but still valid) test.  The angle
between ``S*`` and ``K_rel`` quantifies the price of that robustness.

Nothing here is fitted after seeing data; ``S*`` is fixed by the reference
process and ``K_rel`` by the preregistered noise set.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Sequence

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


def _hs(A: np.ndarray, B: np.ndarray) -> float:
    """Real Hilbert-Schmidt inner product Tr(A^T B) for symmetric operators."""
    return float(np.sum(A * B))


@dataclass
class WitnessCertificate:
    """Dual causal witness of the switch generalized-robustness SDP."""

    R_g: float
    S: np.ndarray
    tightness_gap: float          # |Tr(S W_ref) - R_g|, should be ~0
    equality_residual: float
    status: str
    solver: str = "SCS"

    def value(self, W: np.ndarray) -> float:
        """Witness expectation Tr(S W)."""
        return _hs(self.S, np.asarray(W, dtype=float))


def switch_generalized_robustness_witness(
    W_ref: np.ndarray,
    dims: Sequence[int] = SWITCH_DIMS_WITH_FUTURE,
    solver: str = "SCS",
    eps: float = 1e-9,
    max_iters: int = 200_000,
) -> WitnessCertificate:
    """Return ``R_g`` and the dual-optimal causal witness ``S*`` at ``W_ref``.

    The witness is the symmetrized dual variable of the main equality constraint
    ``W + X = W_AB + W_BA`` of the generalized-robustness program.  By strong
    duality it satisfies ``Tr(S* W_ref) = R_g`` and, because its dual-feasible
    set does not depend on ``W``, ``Tr(S* .)`` lower-bounds ``R_g(.)`` on every
    process (supporting functional at ``W_ref``).
    """
    if cp is None:
        raise RuntimeError("cvxpy is required. Install requirements.txt.")
    dims = tuple(int(d) for d in dims)
    D = 1
    for d in dims:
        D *= d
    d_O = dims[1] * dims[3]
    W = 0.5 * (np.asarray(W_ref, dtype=float) + np.asarray(W_ref, dtype=float).T)

    C_ab = _subspace_complement_rows(L_A_before_B_with_future, dims, "ABF")
    C_ba = _subspace_complement_rows(L_B_before_A_with_future, dims, "BAF")
    C_v = _subspace_complement_rows(L_valid_with_future, dims, "VF")

    W1 = cp.Variable((D, D), symmetric=True)
    W2 = cp.Variable((D, D), symmetric=True)
    X = cp.Variable((D, D), symmetric=True)
    vec = lambda M: cp.reshape(M, (D * D,), order="C")  # noqa: E731
    equality = W + X == W1 + W2
    constraints = [
        W1 >> 0, W2 >> 0, X >> 0,
        C_ab @ vec(W1) == 0, C_ba @ vec(W2) == 0, C_v @ vec(X) == 0,
        equality,
    ]
    problem = cp.Problem(cp.Minimize(cp.trace(X) / d_O), constraints)
    problem.solve(solver="SCS", verbose=False, eps=eps, max_iters=max_iters)

    status = str(problem.status)
    S = 0.5 * (np.asarray(equality.dual_value) + np.asarray(equality.dual_value).T)
    R_g = float(problem.value)
    tightness = abs(_hs(S, W) - R_g)
    eqres = float(np.linalg.norm((W + np.asarray(X.value)) - np.asarray(W1.value) - np.asarray(W2.value)))
    return WitnessCertificate(R_g=R_g, S=S, tightness_gap=tightness,
                              equality_residual=eqres, status=status, solver="SCS")


@dataclass
class AffineCurve:
    """Affine witness signal ``Tr(S W(lambda)) = slope * lambda + intercept``."""

    lambdas: np.ndarray
    values: np.ndarray
    slope: float
    intercept: float
    affinity_residual: float          # max |value - affine fit|, ~0 proves affinity
    zero_crossing: float              # certified threshold lambda* (nan if slope>=0)

    def invert(self, observed_value: float) -> float:
        """Recover lambda from a single scalar expectation (noiseless inverse)."""
        if abs(self.slope) < 1e-15:
            return float("nan")
        return (observed_value - self.intercept) / self.slope


def affine_witness_curve(
    S: np.ndarray,
    family: Callable[[float], np.ndarray],
    lambdas: Sequence[float] | None = None,
) -> AffineCurve:
    """Evaluate the (provably affine) witness signal along an affine family.

    ``family(lambda)`` must be an affine process family (e.g. the partially
    dephased switch).  No SDP is solved here: the signal is pure traces, so this
    is cheap and exact.
    """
    lam = np.asarray(lambdas if lambdas is not None else np.linspace(0.0, 1.0, 11), dtype=float)
    vals = np.array([_hs(S, family(float(x))) for x in lam])
    A = np.vstack([lam, np.ones_like(lam)]).T
    (slope, intercept), *_ = np.linalg.lstsq(A, vals, rcond=None)
    residual = float(np.max(np.abs(vals - (slope * lam + intercept))))
    zero_crossing = float(-intercept / slope) if abs(slope) > 1e-15 and slope < 0 else float("nan")
    return AffineCurve(lambdas=lam, values=vals, slope=float(slope),
                       intercept=float(intercept), affinity_residual=residual,
                       zero_crossing=zero_crossing)


def find_sign_change_crossings(x: np.ndarray, y: np.ndarray) -> list[float]:
    """Linear-interpolated x-locations where y changes sign."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    crossings: list[float] = []
    for i in range(len(x) - 1):
        y0, y1 = y[i], y[i + 1]
        if y0 == 0.0:
            crossings.append(float(x[i]))
        elif y0 * y1 < 0.0:
            t = y0 / (y0 - y1)
            crossings.append(float(x[i] + t * (x[i + 1] - x[i])))
    return crossings


@dataclass
class FamilyCertificate:
    """Certified-witness diagnostics for one process family using fixed ``S*``.

    ``is_affine`` records whether the witness signal is affine in the family
    parameter. The certified region is where ``Tr(S* W) > 0``: a one-sided
    threshold for monotone families, or a two-sided interval otherwise.
    """

    name: str
    params: np.ndarray
    witness_values: np.ndarray
    reference_param: float
    reference_value: float
    is_affine: bool
    affinity_residual: float
    slope: float
    certified_crossings: list
    certified_region: tuple
    single_scalar_injective: bool
    note: str = ""


def certify_family(
    S: np.ndarray,
    family: Callable[[float], np.ndarray],
    params: Sequence[float],
    reference_param: float,
    R_g_ref: float,
) -> FamilyCertificate:
    """Apply the fixed witness ``S*`` to an arbitrary process family.

    The lower-bound property ``Tr(S* W) <= R_g(W)`` holds for any family because
    ``S*`` is dual-feasible. Affinity is an extra property that only holds for
    families affine in the parameter. No SDP is solved here; use the landscape
    script to verify the bound against the full SDP on a grid.
    """
    _ = R_g_ref  # Kept in the API to make the reference benchmark explicit.
    S = np.asarray(S, dtype=float)
    p = np.asarray(params, dtype=float)
    vals = np.array([_hs(S, family(float(x))) for x in p])

    A = np.vstack([p, np.ones_like(p)]).T
    (slope, intercept), *_ = np.linalg.lstsq(A, vals, rcond=None)
    residual = float(np.max(np.abs(vals - (slope * p + intercept))))
    is_affine = residual < 1e-9

    ref_value = float(_hs(S, family(float(reference_param))))
    crossings = find_sign_change_crossings(p, vals)

    lo, hi = float(p.min()), float(p.max())
    below = [c for c in crossings if c < reference_param]
    above = [c for c in crossings if c > reference_param]
    region_lo = max(below) if below else lo
    region_hi = min(above) if above else hi

    diffs = np.diff(vals)
    injective = bool(np.all(diffs > 1e-12) or np.all(diffs < -1e-12))

    note = (
        "affine family: single-scalar inversion is exact and the threshold is one-sided."
        if is_affine and injective
        else "nonlinear/symmetric family: the witness certifies a two-sided interval "
        "and resolves order-coherence magnitude, not its direction."
    )
    return FamilyCertificate(
        name=getattr(family, "__name__", "family"),
        params=p,
        witness_values=vals,
        reference_param=float(reference_param),
        reference_value=ref_value,
        is_affine=is_affine,
        affinity_residual=residual,
        slope=float(slope) if is_affine else float("nan"),
        certified_crossings=[float(c) for c in crossings],
        certified_region=(region_lo, region_hi),
        single_scalar_injective=injective,
        note=note,
    )


@dataclass
class AdmissibleDirection:
    """Relational test direction ``K_rel = P_A(S*)`` and its diagnostics."""

    K_rel: np.ndarray
    cos_angle_to_S: float             # alignment with the raw certificate direction
    noise_leakage: dict               # Tr(K_rel N_i)/||N_i|| per calibrated noise
    signal_response: float            # d<K_rel>/dlambda along the falsification axis
    retained_fraction: float          # ||P_A S*|| / ||S*||


def admissible_direction(
    S: np.ndarray,
    calibrated_noises: Sequence[np.ndarray],
    signal_axis: np.ndarray,
) -> AdmissibleDirection:
    """Project ``S*`` onto the complement of the calibrated-noise subspace.

    ``K_rel`` keeps the signal-detecting component of the certificate while being
    orthogonal (hence first-order insensitive) to every preregistered nuisance
    direction.  This is the physical instantiation of the preregistered
    ``minimum-HS-norm on the admissible face`` rule on the real switch.
    """
    S = np.asarray(S, dtype=float)
    # Orthonormalize the calibrated-noise directions in HS inner product.
    basis: list[np.ndarray] = []
    for N in calibrated_noises:
        N = np.asarray(N, dtype=float)
        v = N.copy()
        for b in basis:
            v = v - _hs(b, v) * b
        nrm = np.sqrt(_hs(v, v))
        if nrm > 1e-12:
            basis.append(v / nrm)
    K = S.copy()
    for b in basis:
        K = K - _hs(b, K) * b  # remove nuisance components
    norm_K = np.sqrt(_hs(K, K))
    norm_S = np.sqrt(_hs(S, S))
    cos_angle = float(_hs(S, K) / (norm_S * norm_K)) if norm_K > 1e-12 else float("nan")
    leakage = {
        f"N{i}": float(_hs(K, np.asarray(N)) / (np.linalg.norm(N) + 1e-30))
        for i, N in enumerate(calibrated_noises)
    }
    signal_response = float(_hs(K, np.asarray(signal_axis, dtype=float)))
    retained = float(norm_K / norm_S) if norm_S > 1e-12 else float("nan")
    return AdmissibleDirection(K_rel=K, cos_angle_to_S=cos_angle,
                              noise_leakage=leakage, signal_response=signal_response,
                              retained_fraction=retained)
