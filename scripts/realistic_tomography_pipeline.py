#!/usr/bin/env python
"""End-to-end realistic tomography simulation for DeltaW/K_rel.

This script is still simulated evidence, not experimental validation.  Its goal
is to close the gap between the old one-parameter smoke test and a usable
finite-count tomography chain:

1. build an informational measurement model M[j, o, k];
2. generate multinomial or Poisson counts with visibility, crosstalk, and drift;
3. reconstruct the coefficient vector by regularized linear inversion or MLE;
4. estimate Fisher and Ledoit-Wolf-style shrinkage covariance;
5. run parametric bootstrap;
6. project onto an admissible direction and report lambda_hat power maps.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

try:
    import cvxpy as cp  # type: ignore
except Exception:  # pragma: no cover
    cp = None


@dataclass(frozen=True)
class TomographyConfig:
    dim: int = 8
    n_settings: int = 24
    n_outcomes: int = 2
    ridge: float = 1e-3
    alpha: float = 0.01
    n_sim: int = 200
    n_null: int = 500
    n_boot: int = 100
    seed: int = 123
    count_model: str = "multinomial"
    reconstruction: str = "linear"


def parse_float_grid(raw: str) -> list[float]:
    values = [float(part.strip()) for part in raw.split(",") if part.strip()]
    if not values:
        raise ValueError("grid must contain at least one numeric value.")
    return values


def parse_int_grid(raw: str) -> list[int]:
    values = [int(part.strip()) for part in raw.split(",") if part.strip()]
    if not values:
        raise ValueError("grid must contain at least one integer value.")
    if any(v <= 0 for v in values):
        raise ValueError("integer grid values must be positive.")
    return values


def _normalize(v: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(v))
    if norm < 1e-12:
        raise ValueError("cannot normalize a near-zero vector.")
    return v / norm


def make_measurement_model(
    n_settings: int,
    n_outcomes: int,
    dim: int,
    rng: np.random.Generator,
    response_scale: float = 0.22,
) -> np.ndarray:
    """Return M[j, o, k] with zero outcome-sum per setting.

    The zero-sum condition keeps per-setting probabilities normalized for any
    coefficient vector before clipping/renormalization.
    """
    if n_settings <= 0 or n_outcomes < 2 or dim <= 0:
        raise ValueError("n_settings>0, n_outcomes>=2 and dim>0 are required.")
    model = rng.normal(size=(n_settings, n_outcomes, dim))
    model -= model.mean(axis=1, keepdims=True)
    max_norm = float(np.max(np.linalg.norm(model.reshape(-1, dim), axis=1)))
    return response_scale * model / max(max_norm, 1e-12)


def make_admissible_basis(dim: int, rng: np.random.Generator, n_vectors: int | None = None) -> np.ndarray:
    """Rows form an orthonormal admissible basis used for projection A."""
    if n_vectors is None:
        n_vectors = max(2, dim // 2)
    if not (1 <= n_vectors <= dim):
        raise ValueError("n_vectors must lie in [1, dim].")
    q, _ = np.linalg.qr(rng.normal(size=(dim, n_vectors)))
    return q.T


def project_admissible(vector: np.ndarray, basis: np.ndarray) -> np.ndarray:
    """Project a coefficient vector onto the row-span of an orthonormal basis."""
    return basis.T @ (basis @ vector)


def drift_pattern(n_settings: int, n_outcomes: int, drift_level: float) -> np.ndarray:
    if drift_level == 0.0:
        return np.zeros((n_settings, n_outcomes))
    setting_axis = np.linspace(-1.0, 1.0, n_settings)
    outcome_axis = np.linspace(-1.0, 1.0, n_outcomes)
    pattern = np.outer(np.sin(np.pi * setting_axis), outcome_axis)
    pattern -= pattern.mean(axis=1, keepdims=True)
    return float(drift_level) * pattern


def probabilities_from_theta(
    model: np.ndarray,
    theta: np.ndarray,
    *,
    visibility: float = 1.0,
    crosstalk: float = 0.0,
    drift_level: float = 0.0,
    eps: float = 1e-9,
) -> np.ndarray:
    """Map coefficients to setting/outcome probabilities."""
    if not (0.0 <= visibility <= 1.0):
        raise ValueError("visibility must lie in [0, 1].")
    if not (0.0 <= crosstalk <= 1.0):
        raise ValueError("crosstalk must lie in [0, 1].")
    n_settings, n_outcomes, dim = model.shape
    theta = np.asarray(theta, dtype=float)
    if theta.shape != (dim,):
        raise ValueError(f"theta must have shape {(dim,)}, got {theta.shape}.")
    base = np.full((n_settings, n_outcomes), 1.0 / n_outcomes)
    signal = visibility * np.tensordot(model, theta, axes=([2], [0]))
    raw = base + signal + drift_pattern(n_settings, n_outcomes, drift_level)
    raw = np.clip(raw, eps, None)
    probs = raw / raw.sum(axis=1, keepdims=True)
    if crosstalk:
        uniform = np.full_like(probs, 1.0 / n_outcomes)
        probs = (1.0 - crosstalk) * probs + crosstalk * uniform
    return probs / probs.sum(axis=1, keepdims=True)


def generate_counts(
    probabilities: np.ndarray,
    shots_per_setting: int,
    rng: np.random.Generator,
    count_model: str = "multinomial",
) -> np.ndarray:
    """Generate finite counts for every setting."""
    if shots_per_setting <= 0:
        raise ValueError("shots_per_setting must be positive.")
    counts = np.zeros_like(probabilities, dtype=int)
    if count_model == "multinomial":
        for j, p in enumerate(probabilities):
            counts[j] = rng.multinomial(shots_per_setting, p)
    elif count_model == "poisson":
        counts = rng.poisson(shots_per_setting * probabilities).astype(int)
    else:
        raise ValueError("count_model must be 'multinomial' or 'poisson'.")
    return counts


def counts_to_frequencies(counts: np.ndarray) -> np.ndarray:
    totals = counts.sum(axis=1, keepdims=True)
    totals = np.maximum(totals, 1)
    return counts / totals


def reconstruct_linear(
    counts: np.ndarray,
    model: np.ndarray,
    ridge: float = 1e-3,
    visibility_assumed: float = 1.0,
) -> np.ndarray:
    """Regularized linear inversion around the uniform baseline."""
    n_settings, n_outcomes, dim = model.shape
    frequencies = counts_to_frequencies(counts)
    y = (frequencies - 1.0 / n_outcomes).reshape(-1)
    X = (visibility_assumed * model).reshape(n_settings * n_outcomes, dim)
    normal = X.T @ X + float(ridge) * np.eye(dim)
    return np.linalg.solve(normal, X.T @ y)


def reconstruct_mle(
    counts: np.ndarray,
    model: np.ndarray,
    ridge: float = 1e-3,
    visibility_assumed: float = 1.0,
    eps: float = 1e-8,
) -> np.ndarray:
    """Constrained multinomial MLE with affine probabilities."""
    if cp is None:
        raise RuntimeError("cvxpy is required for MLE reconstruction.")
    n_settings, n_outcomes, dim = model.shape
    theta = cp.Variable(dim)
    X = (visibility_assumed * model).reshape(n_settings * n_outcomes, dim)
    p = (1.0 / n_outcomes) + X @ theta
    constraints = [p >= eps]
    p_matrix = cp.reshape(p, (n_settings, n_outcomes), order="C")
    constraints += [cp.sum(p_matrix, axis=1) == 1.0]
    counts_flat = counts.reshape(-1).astype(float)
    objective = cp.Minimize(-cp.sum(cp.multiply(counts_flat, cp.log(p))) + ridge * cp.sum_squares(theta))
    problem = cp.Problem(objective, constraints)
    try:
        problem.solve(solver="CLARABEL", verbose=False)
    except Exception:
        problem.solve(solver="SCS", verbose=False, eps=1e-6, max_iters=50_000)
    if problem.status not in {"optimal", "optimal_inaccurate"}:
        raise RuntimeError(f"MLE reconstruction failed with status {problem.status}.")
    return np.asarray(theta.value, dtype=float)


def reconstruct(
    counts: np.ndarray,
    model: np.ndarray,
    *,
    method: str,
    ridge: float,
    visibility_assumed: float,
) -> np.ndarray:
    if method == "linear":
        return reconstruct_linear(counts, model, ridge=ridge, visibility_assumed=visibility_assumed)
    if method == "mle":
        return reconstruct_mle(counts, model, ridge=ridge, visibility_assumed=visibility_assumed)
    raise ValueError("method must be 'linear' or 'mle'.")


def fisher_covariance(
    model: np.ndarray,
    probabilities: np.ndarray,
    shots_per_setting: int,
    ridge: float = 1e-3,
) -> np.ndarray:
    """Approximate covariance from multinomial Fisher information."""
    _, _, dim = model.shape
    fisher = float(ridge) * np.eye(dim)
    for design, p in zip(model, probabilities):
        cov_freq = (np.diag(p) - np.outer(p, p)) / max(shots_per_setting, 1)
        fisher += design.T @ np.linalg.pinv(cov_freq, rcond=1e-10) @ design
    return np.linalg.pinv(fisher, rcond=1e-10)


def ledoit_wolf_shrinkage(samples: np.ndarray) -> tuple[np.ndarray, float]:
    """Ledoit-Wolf-style shrinkage of empirical covariance toward scalar identity."""
    samples = np.asarray(samples, dtype=float)
    if samples.ndim != 2 or samples.shape[0] < 2:
        raise ValueError("samples must have shape (n_samples, dim) with n_samples>=2.")
    n, dim = samples.shape
    centered = samples - samples.mean(axis=0, keepdims=True)
    empirical = np.cov(centered, rowvar=False, ddof=1)
    target = np.eye(dim) * (np.trace(empirical) / dim)
    outer = centered[:, :, None] * centered[:, None, :]
    phi = float(np.mean(np.sum((outer - empirical) ** 2, axis=(1, 2))) / n)
    rho = float(np.sum((empirical - target) ** 2))
    alpha = 1.0 if rho < 1e-15 else float(np.clip(phi / rho, 0.0, 1.0))
    shrunk = alpha * target + (1.0 - alpha) * empirical
    return 0.5 * (shrunk + shrunk.T), alpha


def lambda_hat(theta_hat: np.ndarray, k_rel: np.ndarray, admissible_basis: np.ndarray) -> float:
    projected = project_admissible(theta_hat, admissible_basis)
    denom = float(np.dot(k_rel, k_rel))
    return float(np.dot(projected, k_rel) / max(denom, 1e-15))


def parametric_bootstrap_lambda(
    theta_hat: np.ndarray,
    model: np.ndarray,
    k_rel: np.ndarray,
    admissible_basis: np.ndarray,
    *,
    shots_per_setting: int,
    visibility: float,
    crosstalk: float,
    drift_level: float,
    count_model: str,
    reconstruction: str,
    ridge: float,
    n_boot: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, float]:
    estimates = []
    theta_boot = []
    probs = probabilities_from_theta(
        model, theta_hat, visibility=visibility, crosstalk=crosstalk, drift_level=drift_level
    )
    for _ in range(n_boot):
        counts = generate_counts(probs, shots_per_setting, rng, count_model=count_model)
        boot_hat = reconstruct(
            counts,
            model,
            method=reconstruction,
            ridge=ridge,
            visibility_assumed=visibility,
        )
        theta_boot.append(boot_hat)
        estimates.append(lambda_hat(boot_hat, k_rel, admissible_basis))
    theta_boot_arr = np.asarray(theta_boot)
    shrunk_cov, shrinkage_alpha = ledoit_wolf_shrinkage(theta_boot_arr)
    return np.asarray(estimates), shrunk_cov, shrinkage_alpha


def _single_lambda_estimate(
    model: np.ndarray,
    k_rel: np.ndarray,
    admissible_basis: np.ndarray,
    lambda_true: float,
    shots_per_setting: int,
    visibility: float,
    crosstalk: float,
    drift_level: float,
    count_model: str,
    reconstruction: str,
    ridge: float,
    rng: np.random.Generator,
) -> tuple[float, np.ndarray, np.ndarray]:
    theta_true = float(lambda_true) * k_rel
    probs = probabilities_from_theta(
        model, theta_true, visibility=visibility, crosstalk=crosstalk, drift_level=drift_level
    )
    counts = generate_counts(probs, shots_per_setting, rng, count_model=count_model)
    theta_hat = reconstruct(
        counts,
        model,
        method=reconstruction,
        ridge=ridge,
        visibility_assumed=visibility,
    )
    return lambda_hat(theta_hat, k_rel, admissible_basis), theta_hat, probs


def estimate_power_map(
    config: TomographyConfig,
    *,
    lambda_values: list[float],
    n_total_values: list[int],
    visibility_values: list[float],
    crosstalk_values: list[float],
    drift_values: list[float],
) -> tuple[pd.DataFrame, dict]:
    rng = np.random.default_rng(config.seed)
    model = make_measurement_model(config.n_settings, config.n_outcomes, config.dim, rng)
    admissible_basis = make_admissible_basis(config.dim, rng)
    k_rel = _normalize(project_admissible(rng.normal(size=config.dim), admissible_basis))
    rows = []

    for n_total in n_total_values:
        shots_per_setting = max(1, int(n_total) // config.n_settings)
        for visibility in visibility_values:
            for crosstalk in crosstalk_values:
                for drift_level in drift_values:
                    null_estimates = [
                        _single_lambda_estimate(
                            model,
                            k_rel,
                            admissible_basis,
                            0.0,
                            shots_per_setting,
                            visibility,
                            crosstalk,
                            drift_level,
                            config.count_model,
                            config.reconstruction,
                            config.ridge,
                            rng,
                        )[0]
                        for _ in range(config.n_null)
                    ]
                    null_crit = float(np.quantile(null_estimates, 1.0 - config.alpha))
                    for lam in lambda_values:
                        estimates = []
                        theta_representative = None
                        probs_representative = None
                        for _ in range(config.n_sim):
                            estimate, theta_hat, probs = _single_lambda_estimate(
                                model,
                                k_rel,
                                admissible_basis,
                                lam,
                                shots_per_setting,
                                visibility,
                                crosstalk,
                                drift_level,
                                config.count_model,
                                config.reconstruction,
                                config.ridge,
                                rng,
                            )
                            estimates.append(estimate)
                            theta_representative = theta_hat
                            probs_representative = probs

                        bootstrap_estimates, shrunk_cov, shrinkage_alpha = parametric_bootstrap_lambda(
                            theta_representative,
                            model,
                            k_rel,
                            admissible_basis,
                            shots_per_setting=shots_per_setting,
                            visibility=visibility,
                            crosstalk=crosstalk,
                            drift_level=drift_level,
                            count_model=config.count_model,
                            reconstruction=config.reconstruction,
                            ridge=config.ridge,
                            n_boot=max(2, config.n_boot),
                            rng=rng,
                        )
                        fisher_cov = fisher_covariance(
                            model,
                            probs_representative,
                            shots_per_setting,
                            ridge=config.ridge,
                        )
                        sigma_fisher = math.sqrt(max(0.0, float(k_rel @ fisher_cov @ k_rel)))
                        sigma_shrunk = math.sqrt(max(0.0, float(k_rel @ shrunk_cov @ k_rel)))
                        estimates_arr = np.asarray(estimates)
                        rows.append(
                            {
                                "lambda_true": lam,
                                "n_total": int(n_total),
                                "shots_per_setting": shots_per_setting,
                                "visibility": visibility,
                                "crosstalk": crosstalk,
                                "drift": drift_level,
                                "power": float(np.mean(estimates_arr > null_crit)),
                                "lambda_hat_mean": float(np.mean(estimates_arr)),
                                "lambda_hat_sd": float(np.std(estimates_arr, ddof=1)),
                                "lambda_null_crit": null_crit,
                                "bootstrap_lambda_sd": float(np.std(bootstrap_estimates, ddof=1)),
                                "sigma_lambda_fisher": sigma_fisher,
                                "sigma_lambda_shrunk": sigma_shrunk,
                                "shrinkage_alpha": float(shrinkage_alpha),
                                "n_sim": config.n_sim,
                                "n_null": config.n_null,
                                "n_boot": config.n_boot,
                                "count_model": config.count_model,
                                "reconstruction": config.reconstruction,
                            }
                        )

    metadata = {
        "status": "realistic_tomography_simulation",
        "config": asdict(config),
        "model_shape": list(model.shape),
        "admissible_dim": int(admissible_basis.shape[0]),
        "interpretation": (
            "Finite-count simulated tomography with instrument perturbations. "
            "This is a realistic simulation bridge, not calibrated experimental evidence."
        ),
    }
    return pd.DataFrame(rows), metadata


def _json_safe(obj):
    if isinstance(obj, dict):
        return {k: _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_json_safe(v) for v in obj]
    if isinstance(obj, float) and not math.isfinite(obj):
        return None
    return obj


def write_outputs(df: pd.DataFrame, metadata: dict, outdir: Path) -> tuple[Path, Path, Path]:
    outdir.mkdir(parents=True, exist_ok=True)
    csv_path = outdir / "realistic_tomography_power.csv"
    json_path = outdir / "realistic_tomography_report.json"
    png_path = outdir / "realistic_tomography_power.png"
    df.to_csv(csv_path, index=False)
    report = dict(metadata)
    report["rows"] = df.to_dict(orient="records")
    json_path.write_text(
        json.dumps(_json_safe(report), indent=2, ensure_ascii=False, allow_nan=False),
        encoding="utf-8",
    )
    plot_power(df, png_path)
    return csv_path, json_path, png_path


def plot_power(df: pd.DataFrame, outpath: Path) -> None:
    fig, ax = plt.subplots(figsize=(8.5, 5.2))
    grouped = df[df["lambda_true"] > 0].groupby(["lambda_true", "visibility", "crosstalk", "drift"])
    for (lam, visibility, crosstalk, drift_level), sub in grouped:
        sub = sub.sort_values("n_total")
        label = f"lambda={lam:g}, V={visibility:g}, x={crosstalk:g}, d={drift_level:g}"
        ax.plot(sub["n_total"], sub["power"], marker="o", linewidth=1.5, label=label)
    ax.axhline(0.8, color="#555555", linestyle="--", linewidth=1.0, label="80% target")
    ax.set_xlabel("Total copies / shots")
    ax.set_ylabel("Detection power")
    ax.set_ylim(-0.03, 1.03)
    ax.set_title("Realistic finite-count tomography power map")
    ax.legend(loc="best", fontsize=7)
    fig.tight_layout()
    fig.savefig(outpath, dpi=180)
    plt.close(fig)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--outdir", default="results/realistic_tomography")
    parser.add_argument("--dim", type=int, default=8)
    parser.add_argument("--n-settings", type=int, default=24)
    parser.add_argument("--n-outcomes", type=int, default=2)
    parser.add_argument("--lambda-values", default="0,0.02,0.05")
    parser.add_argument("--n-total-values", default="5000,20000")
    parser.add_argument("--visibility-values", default="0.9,1.0")
    parser.add_argument("--crosstalk-values", default="0,0.02")
    parser.add_argument("--drift-values", default="0,0.01")
    parser.add_argument("--n-sim", type=int, default=200)
    parser.add_argument("--n-null", type=int, default=500)
    parser.add_argument("--n-boot", type=int, default=100)
    parser.add_argument("--ridge", type=float, default=1e-3)
    parser.add_argument("--alpha", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--count-model", choices=["multinomial", "poisson"], default="multinomial")
    parser.add_argument("--reconstruction", choices=["linear", "mle"], default="linear")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config = TomographyConfig(
        dim=args.dim,
        n_settings=args.n_settings,
        n_outcomes=args.n_outcomes,
        ridge=args.ridge,
        alpha=args.alpha,
        n_sim=args.n_sim,
        n_null=args.n_null,
        n_boot=args.n_boot,
        seed=args.seed,
        count_model=args.count_model,
        reconstruction=args.reconstruction,
    )
    df, metadata = estimate_power_map(
        config,
        lambda_values=parse_float_grid(args.lambda_values),
        n_total_values=parse_int_grid(args.n_total_values),
        visibility_values=parse_float_grid(args.visibility_values),
        crosstalk_values=parse_float_grid(args.crosstalk_values),
        drift_values=parse_float_grid(args.drift_values),
    )
    csv_path, json_path, png_path = write_outputs(df, metadata, Path(args.outdir))
    print(f"wrote {csv_path}")
    print(f"wrote {json_path}")
    print(f"wrote {png_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
