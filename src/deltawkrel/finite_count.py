r"""Finite-count statistics of the single-direction estimator (task A4).

Once the falsification test is reduced to one preregistered functional
``g(lambda) = Tr(K W(lambda))`` (with ``K`` either the raw dual witness ``S*``
or the admissible relational direction ``K_rel``), the whole finite-sample
question collapses to *single-parameter* estimation.  This module makes that
quantitative.

Idealisation and honesty
------------------------
We use the **shot-noise floor** of a direct measurement of the witness
observable on the normalised process ``rho(lambda) = W(lambda)/Tr(W)``:

    Var_1(lambda) = Tr(K^2 rho) - Tr(K rho)^2          (per-copy variance)
    Var(lambda_hat) = Var_1 / (N * a^2)                (a = d g_rho / d lambda)

This is a best-case (lower) bound on the achievable variance: a real switch is
characterised by probe-state/measurement tomography, whose calibrated variance
is handled by ``scripts/realistic_tomography_pipeline.py``.  The point of A4 is
the *scaling law* and the *relational payoff*, both of which are already visible
at the shot-noise floor and can only get larger, never smaller, under
calibration.  Nothing here is presented as an experimental forecast.

The key relational result: because ``K_rel`` is orthogonal (in Hilbert-Schmidt
inner product) to the calibrated-noise directions, a calibrated-noise drift does
not move ``Tr(K_rel W)``; it therefore does not inflate the false-positive rate,
whereas the raw witness ``S*`` (not admissible-projected) does.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Sequence

import numpy as np


def _trace(A: np.ndarray, B: np.ndarray) -> float:
    return float(np.sum(A * B))


def normalised_process(W: np.ndarray) -> np.ndarray:
    """rho = W / Tr(W); the witness signal per copy is measured on rho."""
    W = np.asarray(W, dtype=float)
    return W / float(np.trace(W))


def per_copy_variance(K: np.ndarray, W: np.ndarray) -> float:
    """Shot-noise variance Tr(K^2 rho) - Tr(K rho)^2 of observable K on rho(W)."""
    rho = normalised_process(W)
    K = np.asarray(K, dtype=float)
    m1 = _trace(K, rho)
    m2 = float(np.einsum("ij,jk,ki->", K, K, rho))  # Tr(K^2 rho)
    return max(0.0, m2 - m1 * m1)


@dataclass
class ScalingResult:
    slope_rho: float                   # d Tr(K rho)/d lambda along the family
    per_copy_variance_ref: float
    N_grid: np.ndarray
    var_lambda_analytic: np.ndarray
    var_lambda_empirical: np.ndarray


def lambda_estimator_scaling(
    K: np.ndarray,
    family: Callable[[float], np.ndarray],
    lambda_ref: float = 0.1,
    N_grid: Sequence[int] = (100, 300, 1000, 3000, 10000),
    n_repeat: int = 400,
    seed: int = 42,
) -> ScalingResult:
    """Confirm Var(lambda_hat) ~ 1/N by multinomial simulation of the witness.

    The witness observable ``K`` is measured in its eigenbasis on ``rho(lambda)``;
    counts are multinomial with ``N`` copies.  The estimator inverts the affine
    signal ``s(lambda) = Tr(K rho(lambda))``.  Empirical and analytic variances
    are returned for each ``N``.
    """
    rng = np.random.default_rng(seed)
    K = np.asarray(K, dtype=float)
    evals, evecs = np.linalg.eigh(K)

    # affine signal s(lambda) = a*lambda + b -> slope from two points
    s0 = _trace(K, normalised_process(family(0.0)))
    s1 = _trace(K, normalised_process(family(1.0)))
    a = s1 - s0
    b = s0

    rho_ref = normalised_process(family(lambda_ref))
    probs = np.array([float(np.real(v.conj() @ rho_ref @ v)) for v in evecs.T])
    probs = np.clip(probs, 0, None)
    probs = probs / probs.sum()
    v1 = per_copy_variance(K, family(lambda_ref))

    N_grid = np.asarray(list(N_grid), dtype=int)
    var_an = v1 / (N_grid * a * a)
    var_emp = np.zeros_like(N_grid, dtype=float)
    for i, N in enumerate(N_grid):
        lam_hats = np.empty(n_repeat)
        for r in range(n_repeat):
            counts = rng.multinomial(int(N), probs)
            s_hat = float(np.sum(evals * counts) / N)
            lam_hats[r] = (s_hat - b) / a
        var_emp[i] = float(np.var(lam_hats))
    return ScalingResult(slope_rho=float(a), per_copy_variance_ref=float(v1),
                         N_grid=N_grid, var_lambda_analytic=var_an,
                         var_lambda_empirical=var_emp)


def copies_to_certify(
    K: np.ndarray,
    family: Callable[[float], np.ndarray],
    lambda_target: float,
    boundary_value: float = 0.0,
    z_alpha: float = 2.326,          # one-sided 99% (alpha = 0.01)
) -> float:
    """Copies N to certify the witness is on the nonseparable side at alpha.

    Certify ``Tr(K rho(lambda_target)) > boundary`` with confidence ``z_alpha``:

        N >= (z_alpha * sqrt(Var_1) / (s(lambda_target) - boundary))^2 .

    Returns ``inf`` when the signal is at or below the boundary (uncertifiable
    by this single direction at that parameter).
    """
    s = _trace(K, normalised_process(family(lambda_target)))
    margin = s - boundary_value
    if margin <= 0:
        return float("inf")
    v1 = per_copy_variance(K, family(lambda_target))
    return float((z_alpha * np.sqrt(v1) / margin) ** 2)


@dataclass
class FalsePositiveResult:
    drift_grid: np.ndarray
    fp_rate_krel: np.ndarray
    fp_rate_raw: np.ndarray
    krel_signal_shift: np.ndarray
    raw_signal_shift: np.ndarray


def false_positive_under_drift(
    K_rel: np.ndarray,
    S_raw: np.ndarray,
    separable_process: np.ndarray,
    drift_direction: np.ndarray,
    drift_grid: Sequence[float] = (0.0, 0.02, 0.05, 0.1, 0.2),
    N: int = 5000,
    n_trials: int = 400,
    z_alpha: float = 2.326,
    seed: int = 7,
) -> FalsePositiveResult:
    """False-positive rate under calibrated-noise drift on a separable process.

    On a truly causally separable process (witness signal 0), a calibrated-noise
    drift ``delta`` is added along a nuisance direction.  ``K_rel`` is orthogonal
    to that direction, so its signal does not move -> FP rate stays near alpha.
    The raw witness ``S*`` is not orthogonal, so its signal drifts positive ->
    FP inflation.  This is the quantitative payoff of the admissible projection.
    """
    rng = np.random.default_rng(seed)
    drift_grid = np.asarray(list(drift_grid), dtype=float)
    nuis = np.asarray(drift_direction, dtype=float)

    # Baseline (physical, PSD) per-copy variances set the shot-noise scale; a
    # calibrated drift moves the reconstructed process (and hence the witness
    # value) but not the shot-noise scale to first order.  We therefore use the
    # asymptotic-normal witness estimator g_hat ~ N(Tr(K W), v_base/N), which is
    # the standard model for a linear functional of a tomographic reconstruction
    # and is well defined even when the drifted reconstruction leaves the cone.
    v_base_k = max(1e-12, per_copy_variance(K_rel, separable_process))
    v_base_r = max(1e-12, per_copy_variance(S_raw, separable_process))
    se_k, se_r = np.sqrt(v_base_k / N), np.sqrt(v_base_r / N)

    def _fp(K, se, delta):
        g_true = _trace(K, normalised_process(separable_process + delta * nuis))
        samples = rng.normal(g_true, se, size=n_trials)
        return float(np.mean(samples - z_alpha * se > 0.0))   # falsely certify nonseparable

    fp_k, fp_r, sh_k, sh_r = [], [], [], []
    for delta in drift_grid:
        W = separable_process + delta * nuis
        sh_k.append(_trace(K_rel, normalised_process(W)))
        sh_r.append(_trace(S_raw, normalised_process(W)))
        fp_k.append(_fp(K_rel, se_k, delta))
        fp_r.append(_fp(S_raw, se_r, delta))
    return FalsePositiveResult(
        drift_grid=drift_grid,
        fp_rate_krel=np.array(fp_k), fp_rate_raw=np.array(fp_r),
        krel_signal_shift=np.array(sh_k), raw_signal_shift=np.array(sh_r),
    )
