#!/usr/bin/env python3
"""Simplified micro-tomography stress-test for DeltaW/K_rel.

This is NOT full process-matrix tomography. It simulates a whitened one-parameter
micro-tomography problem with finite binomial counts across informational
configurations. It is intended as a reproducible proof of concept and as an
upper-bound / stress-test bridge between the toy LR model and a complete
SDP + MLE reconstruction pipeline.
"""
from __future__ import annotations

import argparse
from pathlib import Path
from dataclasses import dataclass, asdict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


@dataclass
class Result:
    visibility: float
    n_total: int
    lambda_true: float
    power: float
    lr_crit: float
    n_sim: int
    n_null: int
    n_config: int
    seed: int


def make_design(n_config: int, rng: np.random.Generator) -> np.ndarray:
    """Create a normalized pseudo-tomographic design vector."""
    k = rng.normal(size=n_config)
    k -= k.mean()
    k /= np.linalg.norm(k)
    return k


def simulate_lr(lambda_true: float, visibility: float, n_total: int, k: np.ndarray,
                n_rep: int, rng: np.random.Generator) -> np.ndarray:
    """Simulate LR statistics for a one-sided λ>=0 test in a binomial micro-model."""
    n_config = len(k)
    shots = max(1, n_total // n_config)
    # Map the directional signal into probabilities; clipping is a safety guard.
    p = 0.5 + 0.5 * visibility * lambda_true * k / max(np.max(np.abs(k)), 1e-12)
    p = np.clip(p, 1e-6, 1 - 1e-6)
    counts = rng.binomial(shots, p, size=(n_rep, n_config))
    freq = counts / shots
    # Whitened residual around null p0=0.5.
    y = freq - 0.5
    # Under p≈0.5, Var(freq)=0.25/shots.
    weight = 4.0 * shots
    denom = weight * np.dot(k, k)
    lam_hat = (weight * (y @ k)) / denom
    lam_hat_pos = np.maximum(0.0, lam_hat)
    # LR improvement between null and one-parameter fit.
    lr = weight * (2 * visibility * lam_hat_pos) ** 2 * np.dot(k, k)
    return lr


def estimate_power(lambda_values, visibilities, n_values, n_config, n_sim, n_null, seed):
    rng = np.random.default_rng(seed)
    k = make_design(n_config, rng)
    rows = []
    for n_total in n_values:
        for visibility in visibilities:
            lr_null = simulate_lr(0.0, visibility, n_total, k, n_null, rng)
            lr_crit = float(np.quantile(lr_null, 0.99))
            for lam in lambda_values:
                lr = simulate_lr(lam, visibility, n_total, k, n_sim, rng)
                power = float(np.mean(lr > lr_crit))
                rows.append(Result(visibility, n_total, lam, power, lr_crit, n_sim, n_null, n_config, seed))
    return pd.DataFrame([asdict(r) for r in rows])


def plot_power(df: pd.DataFrame, outpath: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    for (n_total, lam), sub in df.groupby(["n_total", "lambda_true"]):
        if lam == 0:
            continue
        label = f"λ={lam:g}, N={n_total:g}"
        sub = sub.sort_values("visibility")
        ax.plot(sub["visibility"], sub["power"], marker="o", label=label)
    ax.axhline(0.8, linestyle="--", linewidth=1, label="80% target")
    ax.set_xlabel("Visibility")
    ax.set_ylabel("Power")
    ax.set_ylim(0, 1.05)
    ax.set_title("Simplified micro-tomography stress-test")
    ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    fig.savefig(outpath, dpi=180)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--outdir", default="outputs")
    parser.add_argument("--n-config", type=int, default=64)
    parser.add_argument("--n-sim", type=int, default=1000)
    parser.add_argument("--n-null", type=int, default=3000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    df = estimate_power(
        lambda_values=[0.0, 0.003, 0.005, 0.010],
        visibilities=[0.95, 0.97, 0.99],
        n_values=[100_000],
        n_config=args.n_config,
        n_sim=args.n_sim,
        n_null=args.n_null,
        seed=args.seed,
    )
    csv_path = outdir / "micro_tomography_power.csv"
    png_path = outdir / "power_micro_tomography.png"
    df.to_csv(csv_path, index=False)
    plot_power(df, png_path)
    print(f"Wrote {csv_path}")
    print(f"Wrote {png_path}")


if __name__ == "__main__":
    main()
