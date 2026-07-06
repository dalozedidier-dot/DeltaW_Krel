from __future__ import annotations

import json

import numpy as np
import pytest

import realistic_tomography_pipeline as tomo


def test_measurement_model_probabilities_and_counts_are_valid():
    rng = np.random.default_rng(1)
    model = tomo.make_measurement_model(5, 3, 4, rng)
    theta = np.array([0.1, -0.05, 0.02, 0.03])
    probs = tomo.probabilities_from_theta(
        model, theta, visibility=0.9, crosstalk=0.03, drift_level=0.01
    )
    assert probs.shape == (5, 3)
    assert np.allclose(probs.sum(axis=1), 1.0)
    assert np.all(probs > 0.0)
    counts = tomo.generate_counts(probs, 100, rng, count_model="multinomial")
    assert counts.shape == probs.shape
    assert np.all(counts.sum(axis=1) == 100)
    poisson = tomo.generate_counts(probs, 100, rng, count_model="poisson")
    assert poisson.shape == probs.shape


def test_linear_reconstruction_recovers_direction_in_noiseless_large_count_limit():
    rng = np.random.default_rng(2)
    model = tomo.make_measurement_model(12, 2, 5, rng)
    basis = tomo.make_admissible_basis(5, rng, n_vectors=3)
    k_rel = tomo._normalize(tomo.project_admissible(rng.normal(size=5), basis))
    theta = 0.25 * k_rel
    probs = tomo.probabilities_from_theta(model, theta)
    counts = np.round(200_000 * probs).astype(int)
    theta_hat = tomo.reconstruct_linear(counts, model, ridge=1e-8)
    estimate = tomo.lambda_hat(theta_hat, k_rel, basis)
    assert estimate == pytest.approx(0.25, abs=0.03)


def test_mle_reconstruction_runs_on_small_problem():
    pytest.importorskip("cvxpy")
    rng = np.random.default_rng(3)
    model = tomo.make_measurement_model(4, 2, 3, rng)
    theta = np.array([0.05, -0.02, 0.03])
    probs = tomo.probabilities_from_theta(model, theta)
    counts = tomo.generate_counts(probs, 200, rng)
    theta_hat = tomo.reconstruct_mle(counts, model, ridge=1e-3)
    assert theta_hat.shape == theta.shape
    assert np.all(np.isfinite(theta_hat))


def test_fisher_and_shrinkage_covariances_are_psd():
    rng = np.random.default_rng(4)
    model = tomo.make_measurement_model(6, 2, 4, rng)
    probs = tomo.probabilities_from_theta(model, np.zeros(4))
    fisher_cov = tomo.fisher_covariance(model, probs, shots_per_setting=100, ridge=1e-3)
    assert np.min(np.linalg.eigvalsh(fisher_cov)) >= -1e-10
    samples = rng.normal(size=(30, 4))
    shrunk, alpha = tomo.ledoit_wolf_shrinkage(samples)
    assert 0.0 <= alpha <= 1.0
    assert np.min(np.linalg.eigvalsh(shrunk)) >= -1e-10


def test_estimate_power_map_and_outputs_are_strict(tmp_path):
    config = tomo.TomographyConfig(
        dim=4,
        n_settings=6,
        n_outcomes=2,
        n_sim=8,
        n_null=12,
        n_boot=5,
        seed=5,
    )
    df, metadata = tomo.estimate_power_map(
        config,
        lambda_values=[0.0, 0.05],
        n_total_values=[600],
        visibility_values=[0.95],
        crosstalk_values=[0.0],
        drift_values=[0.0],
    )
    assert len(df) == 2
    assert set(df.columns) >= {
        "power",
        "lambda_hat_mean",
        "bootstrap_lambda_sd",
        "sigma_lambda_fisher",
        "sigma_lambda_shrunk",
        "shrinkage_alpha",
    }
    assert df["power"].between(0.0, 1.0).all()
    csv_path, json_path, png_path = tomo.write_outputs(df, metadata, tmp_path)
    assert csv_path.exists()
    assert png_path.exists()
    text = json_path.read_text(encoding="utf-8")
    assert "NaN" not in text
    report = json.loads(text)
    assert report["status"] == "realistic_tomography_simulation"
    assert len(report["rows"]) == 2


def test_cli_main_writes_outputs(tmp_path):
    code = tomo.main(
        [
            "--outdir",
            str(tmp_path),
            "--dim",
            "4",
            "--n-settings",
            "5",
            "--lambda-values",
            "0,0.05",
            "--n-total-values",
            "500",
            "--visibility-values",
            "1",
            "--crosstalk-values",
            "0",
            "--drift-values",
            "0",
            "--n-sim",
            "5",
            "--n-null",
            "8",
            "--n-boot",
            "4",
        ]
    )
    assert code == 0
    assert (tmp_path / "realistic_tomography_power.csv").exists()
    assert (tmp_path / "realistic_tomography_report.json").exists()
