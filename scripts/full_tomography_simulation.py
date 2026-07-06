#!/usr/bin/env python
"""Full end-to-end simulated tomography stress test for DeltaW/K_rel.

This is the heaviest simulated test layer in the repository.  It composes the
realistic tomography primitives with physical-noise knobs that matter for an
experimental quantum-switch protocol: finite counts, visibility, control
dephasing, path-dependent losses, crosstalk, temporal drift, optional
interleaved control correction, covariance shrinkage, bootstrap uncertainty,
empirical LR calibration, and an applicability map.

The output is still simulated evidence.  It should be read as a feasibility and
pipeline-stress artifact until the measurement tensor and noise parameters are
calibrated from a real experiment.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "matplotlib-deltawkrel"))
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

_SCRIPTS = Path(__file__).resolve().parent
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

import realistic_tomography_pipeline as tomo


@dataclass(frozen=True)
class FullTomographyConfig:
    dim: int = 8
    n_settings: int = 24
    n_outcomes: int = 2
    ridge: float = 1e-3
    alpha: float = 0.01
    n_sim: int = 200
    n_null: int = 500
    n_boot: int = 100
    seed: int = 321
    count_model: str = "multinomial"
    reconstruction: str = "linear"
    correction: str = "interleaved"
    eps_direction: float = 1e-10
    max_miscalibrated_fpr: float = 0.05


def _unit_interval(value: float, name: str, *, upper_open: bool = False) -> float:
    x = float(value)
    ok = 0.0 <= x < 1.0 if upper_open else 0.0 <= x <= 1.0
    if not ok:
        bracket = "[0, 1)" if upper_open else "[0, 1]"
        raise ValueError(f"{name} must lie in {bracket}.")
    return x


def effective_visibility(visibility: float, dephasing: float, operation_crosstalk: float) -> float:
    """Visibility remaining after coherent dephasing and operation crosstalk."""
    visibility = _unit_interval(visibility, "visibility")
    dephasing = _unit_interval(dephasing, "dephasing")
    operation_crosstalk = _unit_interval(operation_crosstalk, "operation_crosstalk")
    return visibility * (1.0 - dephasing) * (1.0 - operation_crosstalk)


def effective_crosstalk(control_crosstalk: float, operation_crosstalk: float) -> float:
    """Combine independent crosstalk channels into one uniform-mixing rate."""
    control = _unit_interval(control_crosstalk, "control_crosstalk")
    operation = _unit_interval(operation_crosstalk, "operation_crosstalk")
    return 1.0 - (1.0 - control) * (1.0 - operation)


def path_loss_shots(n_total: int, n_settings: int, path_loss: float) -> np.ndarray:
    """Deterministic path-dependent shot schedule with total loss."""
    if n_total <= 0 or n_settings <= 0:
        raise ValueError("n_total and n_settings must be positive.")
    loss = _unit_interval(path_loss, "path_loss", upper_open=True)
    base = max(1, int(n_total) // int(n_settings))
    phase = np.linspace(0.0, 2.0 * np.pi, int(n_settings), endpoint=False)
    loss_profile = loss * (0.5 + 0.5 * np.sin(phase))
    shots = np.maximum(1, np.round(base * (1.0 - loss_profile))).astype(int)
    return shots


def generate_variable_counts(
    probabilities: np.ndarray,
    shots_by_setting: np.ndarray,
    rng: np.random.Generator,
    count_model: str,
) -> np.ndarray:
    counts = np.zeros_like(probabilities, dtype=int)
    for j, shots in enumerate(shots_by_setting):
        counts[j] = tomo.generate_counts(
            probabilities[j : j + 1],
            int(shots),
            rng,
            count_model=count_model,
        )[0]
    return counts


def reconstruct_linear_with_baseline(
    counts: np.ndarray,
    baseline_frequencies: np.ndarray,
    model: np.ndarray,
    ridge: float,
    visibility_assumed: float,
) -> np.ndarray:
    """Linear inversion using interleaved controls as per-setting baseline."""
    n_settings, n_outcomes, dim = model.shape
    frequencies = tomo.counts_to_frequencies(counts)
    y = (frequencies - baseline_frequencies).reshape(-1)
    X = (visibility_assumed * model).reshape(n_settings * n_outcomes, dim)
    normal = X.T @ X + float(ridge) * np.eye(dim)
    return np.linalg.solve(normal, X.T @ y)


def simulate_estimate(
    model: np.ndarray,
    k_rel: np.ndarray,
    admissible_basis: np.ndarray,
    *,
    lambda_true: float,
    shots_by_setting: np.ndarray,
    visibility: float,
    dephasing: float,
    control_crosstalk: float,
    operation_crosstalk: float,
    drift: float,
    correction: str,
    count_model: str,
    reconstruction: str,
    ridge: float,
    rng: np.random.Generator,
) -> tuple[float, np.ndarray, np.ndarray]:
    vis_eff = effective_visibility(visibility, dephasing, operation_crosstalk)
    cross_eff = effective_crosstalk(control_crosstalk, operation_crosstalk)
    theta_true = float(lambda_true) * k_rel
    probs = tomo.probabilities_from_theta(
        model,
        theta_true,
        visibility=vis_eff,
        crosstalk=cross_eff,
        drift_level=drift,
    )
    counts = generate_variable_counts(probs, shots_by_setting, rng, count_model)

    if correction == "none":
        theta_hat = tomo.reconstruct(
            counts,
            model,
            method=reconstruction,
            ridge=ridge,
            visibility_assumed=max(vis_eff, 1e-9),
        )
    elif correction == "interleaved":
        if reconstruction != "linear":
            raise ValueError("interleaved correction currently requires linear reconstruction.")
        control_probs = tomo.probabilities_from_theta(
            model,
            np.zeros(model.shape[2]),
            visibility=vis_eff,
            crosstalk=cross_eff,
            drift_level=drift,
        )
        control_counts = generate_variable_counts(control_probs, shots_by_setting, rng, count_model)
        theta_hat = reconstruct_linear_with_baseline(
            counts,
            tomo.counts_to_frequencies(control_counts),
            model,
            ridge,
            max(vis_eff, 1e-9),
        )
    else:
        raise ValueError("correction must be 'none' or 'interleaved'.")

    return tomo.lambda_hat(theta_hat, k_rel, admissible_basis), theta_hat, probs


def likelihood_ratio(estimates: np.ndarray, null_sd: float) -> np.ndarray:
    scale = max(float(null_sd), 1e-12)
    return (np.maximum(0.0, np.asarray(estimates, dtype=float)) / scale) ** 2


def applicability_record(
    *,
    projected_direction_norm: float,
    null_sd: float,
    lambda_true: float,
    visibility_eff: float,
    false_positive_miscalibrated: float,
    config: FullTomographyConfig,
) -> dict:
    signal_to_noise = visibility_eff * abs(float(lambda_true)) / max(float(null_sd), 1e-12)
    passed = (
        projected_direction_norm > config.eps_direction
        and np.isfinite(null_sd)
        and null_sd > 0.0
        and np.isfinite(signal_to_noise)
        and false_positive_miscalibrated <= config.max_miscalibrated_fpr
    )
    return {
        "projected_direction_norm": projected_direction_norm,
        "null_sd": null_sd,
        "effective_visibility": visibility_eff,
        "signal_to_noise_proxy": signal_to_noise,
        "false_positive_miscalibrated": false_positive_miscalibrated,
        "applicability_passed": bool(passed),
    }


def estimate_full_map(
    config: FullTomographyConfig,
    *,
    lambda_values: list[float],
    n_total_values: list[int],
    visibility_values: list[float],
    dephasing_values: list[float],
    control_crosstalk_values: list[float],
    operation_crosstalk_values: list[float],
    drift_values: list[float],
    path_loss_values: list[float],
) -> tuple[pd.DataFrame, dict]:
    rng = np.random.default_rng(config.seed)
    model = tomo.make_measurement_model(config.n_settings, config.n_outcomes, config.dim, rng)
    admissible_basis = tomo.make_admissible_basis(config.dim, rng)
    k_rel = tomo._normalize(tomo.project_admissible(rng.normal(size=config.dim), admissible_basis))
    projected_direction_norm = float(np.linalg.norm(tomo.project_admissible(k_rel, admissible_basis)))
    rows: list[dict] = []

    for n_total in n_total_values:
        for visibility in visibility_values:
            for dephasing in dephasing_values:
                for control_crosstalk in control_crosstalk_values:
                    for operation_crosstalk in operation_crosstalk_values:
                        for drift in drift_values:
                            for path_loss in path_loss_values:
                                shots = path_loss_shots(n_total, config.n_settings, path_loss)
                                vis_eff = effective_visibility(
                                    visibility, dephasing, operation_crosstalk
                                )
                                cross_eff = effective_crosstalk(
                                    control_crosstalk, operation_crosstalk
                                )
                                null_estimates = np.asarray(
                                    [
                                        simulate_estimate(
                                            model,
                                            k_rel,
                                            admissible_basis,
                                            lambda_true=0.0,
                                            shots_by_setting=shots,
                                            visibility=visibility,
                                            dephasing=dephasing,
                                            control_crosstalk=control_crosstalk,
                                            operation_crosstalk=operation_crosstalk,
                                            drift=drift,
                                            correction=config.correction,
                                            count_model=config.count_model,
                                            reconstruction=config.reconstruction,
                                            ridge=config.ridge,
                                            rng=rng,
                                        )[0]
                                        for _ in range(config.n_null)
                                    ]
                                )
                                null_sd = float(np.std(null_estimates, ddof=1))
                                lr_null = likelihood_ratio(null_estimates, null_sd)
                                lr_crit = float(np.quantile(lr_null, 1.0 - config.alpha))

                                calibrated_null = np.asarray(
                                    [
                                        simulate_estimate(
                                            model,
                                            k_rel,
                                            admissible_basis,
                                            lambda_true=0.0,
                                            shots_by_setting=shots,
                                            visibility=visibility,
                                            dephasing=dephasing,
                                            control_crosstalk=control_crosstalk,
                                            operation_crosstalk=operation_crosstalk,
                                            drift=0.0,
                                            correction=config.correction,
                                            count_model=config.count_model,
                                            reconstruction=config.reconstruction,
                                            ridge=config.ridge,
                                            rng=rng,
                                        )[0]
                                        for _ in range(config.n_null)
                                    ]
                                )
                                lr_calibrated = likelihood_ratio(
                                    calibrated_null,
                                    float(np.std(calibrated_null, ddof=1)),
                                )
                                lr_crit_miscalibrated = float(
                                    np.quantile(lr_calibrated, 1.0 - config.alpha)
                                )
                                false_positive_miscalibrated = float(
                                    np.mean(lr_null > lr_crit_miscalibrated)
                                )

                                for lam in lambda_values:
                                    estimates = []
                                    theta_rep = None
                                    probs_rep = None
                                    for _ in range(config.n_sim):
                                        estimate, theta_hat, probs = simulate_estimate(
                                            model,
                                            k_rel,
                                            admissible_basis,
                                            lambda_true=lam,
                                            shots_by_setting=shots,
                                            visibility=visibility,
                                            dephasing=dephasing,
                                            control_crosstalk=control_crosstalk,
                                            operation_crosstalk=operation_crosstalk,
                                            drift=drift,
                                            correction=config.correction,
                                            count_model=config.count_model,
                                            reconstruction=config.reconstruction,
                                            ridge=config.ridge,
                                            rng=rng,
                                        )
                                        estimates.append(estimate)
                                        theta_rep = theta_hat
                                        probs_rep = probs

                                    estimates_arr = np.asarray(estimates)
                                    lr_alt = likelihood_ratio(estimates_arr, null_sd)
                                    boot, shrunk_cov, shrinkage_alpha = tomo.parametric_bootstrap_lambda(
                                        theta_rep,
                                        model,
                                        k_rel,
                                        admissible_basis,
                                        shots_per_setting=max(1, int(np.mean(shots))),
                                        visibility=vis_eff,
                                        crosstalk=cross_eff,
                                        drift_level=drift,
                                        count_model=config.count_model,
                                        reconstruction=config.reconstruction,
                                        ridge=config.ridge,
                                        n_boot=max(2, config.n_boot),
                                        rng=rng,
                                    )
                                    fisher_cov = tomo.fisher_covariance(
                                        model,
                                        probs_rep,
                                        max(1, int(np.mean(shots))),
                                        ridge=config.ridge,
                                    )
                                    applicability = applicability_record(
                                        projected_direction_norm=projected_direction_norm,
                                        null_sd=null_sd,
                                        lambda_true=lam,
                                        visibility_eff=vis_eff,
                                        false_positive_miscalibrated=false_positive_miscalibrated,
                                        config=config,
                                    )
                                    rows.append(
                                        {
                                            "lambda_true": lam,
                                            "n_total": int(n_total),
                                            "mean_shots_per_setting": float(np.mean(shots)),
                                            "min_shots_per_setting": int(np.min(shots)),
                                            "visibility": visibility,
                                            "dephasing": dephasing,
                                            "control_crosstalk": control_crosstalk,
                                            "operation_crosstalk": operation_crosstalk,
                                            "effective_visibility": vis_eff,
                                            "effective_crosstalk": cross_eff,
                                            "drift": drift,
                                            "path_loss": path_loss,
                                            "correction": config.correction,
                                            "lr_crit": lr_crit,
                                            "power": float(np.mean(lr_alt > lr_crit)),
                                            "false_positive_miscalibrated": false_positive_miscalibrated,
                                            "lambda_hat_mean": float(np.mean(estimates_arr)),
                                            "lambda_hat_sd": float(np.std(estimates_arr, ddof=1)),
                                            "bootstrap_lambda_sd": float(np.std(boot, ddof=1)),
                                            "sigma_lambda_fisher": math.sqrt(
                                                max(0.0, float(k_rel @ fisher_cov @ k_rel))
                                            ),
                                            "sigma_lambda_shrunk": math.sqrt(
                                                max(0.0, float(k_rel @ shrunk_cov @ k_rel))
                                            ),
                                            "shrinkage_alpha": float(shrinkage_alpha),
                                            **applicability,
                                        }
                                    )

    metadata = {
        "status": "full_tomography_simulation",
        "config": asdict(config),
        "model_shape": list(model.shape),
        "rows": len(rows),
        "interpretation": (
            "Full simulated tomography stress test with physical-noise knobs. "
            "This is not calibrated experimental evidence."
        ),
    }
    return pd.DataFrame(rows), metadata


def _json_safe(obj):
    return tomo._json_safe(obj)


def plot_full_map(df: pd.DataFrame, outpath: Path) -> None:
    fig, ax = plt.subplots(figsize=(9.0, 5.4))
    subset = df[df["lambda_true"] > 0].copy()
    if subset.empty:
        subset = df.copy()
    grouped = subset.groupby(["lambda_true", "visibility", "drift", "path_loss"])
    for (lam, visibility, drift, path_loss), sub in grouped:
        sub = sub.sort_values("n_total")
        label = f"lambda={lam:g}, V={visibility:g}, drift={drift:g}, loss={path_loss:g}"
        ax.plot(sub["n_total"], sub["power"], marker="o", linewidth=1.4, label=label)
    ax.axhline(0.8, color="#555555", linestyle="--", linewidth=1.0, label="80% target")
    ax.set_xlabel("Total nominal copies / shots")
    ax.set_ylabel("Empirical LR power")
    ax.set_ylim(-0.03, 1.03)
    ax.set_title("Full simulated tomography stress test")
    ax.legend(loc="best", fontsize=7)
    fig.tight_layout()
    fig.savefig(outpath, dpi=180)
    plt.close(fig)


def write_outputs(df: pd.DataFrame, metadata: dict, outdir: Path) -> tuple[Path, Path, Path]:
    outdir.mkdir(parents=True, exist_ok=True)
    csv_path = outdir / "full_tomography_power.csv"
    json_path = outdir / "full_tomography_report.json"
    png_path = outdir / "full_tomography_power.png"
    df.to_csv(csv_path, index=False)
    report = dict(metadata)
    report["rows"] = df.to_dict(orient="records")
    json_path.write_text(
        json.dumps(_json_safe(report), indent=2, ensure_ascii=False, allow_nan=False),
        encoding="utf-8",
    )
    plot_full_map(df, png_path)
    return csv_path, json_path, png_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--outdir", default="results/full_tomography")
    parser.add_argument("--dim", type=int, default=8)
    parser.add_argument("--n-settings", type=int, default=24)
    parser.add_argument("--n-outcomes", type=int, default=2)
    parser.add_argument("--lambda-values", default="0,0.02,0.05")
    parser.add_argument("--n-total-values", default="5000,20000")
    parser.add_argument("--visibility-values", default="0.9,1.0")
    parser.add_argument("--dephasing-values", default="0,0.05")
    parser.add_argument("--control-crosstalk-values", default="0,0.01")
    parser.add_argument("--operation-crosstalk-values", default="0,0.01")
    parser.add_argument("--drift-values", default="0,0.005")
    parser.add_argument("--path-loss-values", default="0,0.05")
    parser.add_argument("--n-sim", type=int, default=200)
    parser.add_argument("--n-null", type=int, default=500)
    parser.add_argument("--n-boot", type=int, default=100)
    parser.add_argument("--ridge", type=float, default=1e-3)
    parser.add_argument("--alpha", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=321)
    parser.add_argument("--count-model", choices=["multinomial", "poisson"], default="multinomial")
    parser.add_argument("--reconstruction", choices=["linear", "mle"], default="linear")
    parser.add_argument("--correction", choices=["none", "interleaved"], default="interleaved")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config = FullTomographyConfig(
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
        correction=args.correction,
    )
    df, metadata = estimate_full_map(
        config,
        lambda_values=tomo.parse_float_grid(args.lambda_values),
        n_total_values=tomo.parse_int_grid(args.n_total_values),
        visibility_values=tomo.parse_float_grid(args.visibility_values),
        dephasing_values=tomo.parse_float_grid(args.dephasing_values),
        control_crosstalk_values=tomo.parse_float_grid(args.control_crosstalk_values),
        operation_crosstalk_values=tomo.parse_float_grid(args.operation_crosstalk_values),
        drift_values=tomo.parse_float_grid(args.drift_values),
        path_loss_values=tomo.parse_float_grid(args.path_loss_values),
    )
    csv_path, json_path, png_path = write_outputs(df, metadata, Path(args.outdir))
    print(f"wrote {csv_path}")
    print(f"wrote {json_path}")
    print(f"wrote {png_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
