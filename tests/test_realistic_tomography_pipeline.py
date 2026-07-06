from __future__ import annotations

import json

import numpy as np
import pytest

import realistic_tomography_pipeline as tomo


def test_measurement_model_probabilities_and_counts_are_valid():
    rng = np.random.default_rng(1)
    model = tomo.make_measurement_model(5, 3, 4, rng)
    assert np.allclose(model.sum(axis=1), 0.0)
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


def test_input_validation_and_error_paths_are_explicit():
    rng = np.random.default_rng(11)
    with pytest.raises(ValueError):
        tomo.parse_float_grid("")
    with pytest.raises(ValueError):
        tomo.parse_int_grid("")
    with pytest.raises(ValueError):
        tomo.parse_int_grid("0,10")
    with pytest.raises(ValueError):
        tomo._normalize(np.zeros(3))
    with pytest.raises(ValueError):
        tomo.make_measurement_model(0, 2, 4, rng)
    with pytest.raises(ValueError):
        tomo.make_admissible_basis(4, rng, n_vectors=5)

    model = tomo.make_measurement_model(3, 2, 4, rng)
    with pytest.raises(ValueError):
        tomo.probabilities_from_theta(model, np.zeros(4), visibility=1.2)
    with pytest.raises(ValueError):
        tomo.probabilities_from_theta(model, np.zeros(4), crosstalk=-0.1)
    with pytest.raises(ValueError):
        tomo.probabilities_from_theta(model, np.zeros(3))
    probs = tomo.probabilities_from_theta(model, np.zeros(4))
    with pytest.raises(ValueError):
        tomo.generate_counts(probs, 0, rng)
    with pytest.raises(ValueError):
        tomo.generate_counts(probs, 10, rng, count_model="invalid")


def test_visibility_and_crosstalk_move_probabilities_toward_uniform():
    rng = np.random.default_rng(12)
    model = tomo.make_measurement_model(8, 2, 4, rng)
    theta = tomo._normalize(rng.normal(size=4)) * 0.2
    uniform = np.full((8, 2), 0.5)
    p_visible = tomo.probabilities_from_theta(model, theta, visibility=1.0, crosstalk=0.0)
    p_hidden = tomo.probabilities_from_theta(model, theta, visibility=0.0, crosstalk=0.0)
    p_cross = tomo.probabilities_from_theta(model, theta, visibility=1.0, crosstalk=0.5)
    assert np.allclose(p_hidden, uniform)
    assert np.linalg.norm(p_cross - uniform) < np.linalg.norm(p_visible - uniform)
    assert np.allclose(tomo.drift_pattern(5, 2, 0.0), 0.0)
    drift = tomo.drift_pattern(5, 2, 0.03)
    assert np.allclose(drift.sum(axis=1), 0.0)


def test_counts_to_frequencies_handles_empty_rows():
    counts = np.array([[0, 0], [2, 3]])
    freqs = tomo.counts_to_frequencies(counts)
    assert np.allclose(freqs[0], [0.0, 0.0])
    assert np.allclose(freqs[1], [0.4, 0.6])


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


def test_reconstruct_dispatch_and_mle_missing_dependency(monkeypatch):
    rng = np.random.default_rng(13)
    model = tomo.make_measurement_model(4, 2, 3, rng)
    counts = tomo.generate_counts(tomo.probabilities_from_theta(model, np.zeros(3)), 50, rng)
    linear = tomo.reconstruct(counts, model, method="linear", ridge=1e-3, visibility_assumed=1.0)
    assert linear.shape == (3,)
    with pytest.raises(ValueError):
        tomo.reconstruct(counts, model, method="unknown", ridge=1e-3, visibility_assumed=1.0)

    monkeypatch.setattr(tomo, "cp", None)
    with pytest.raises(RuntimeError):
        tomo.reconstruct_mle(counts, model)


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


def test_shrinkage_and_json_safe_cover_degenerate_cases():
    with pytest.raises(ValueError):
        tomo.ledoit_wolf_shrinkage(np.ones((1, 3)))
    shrunk, alpha = tomo.ledoit_wolf_shrinkage(np.ones((4, 3)))
    assert alpha == 1.0
    assert np.allclose(shrunk, 0.0)
    safe = tomo._json_safe({"x": [float("nan"), float("inf"), 1.0]})
    assert safe == {"x": [None, None, 1.0]}


def test_parametric_bootstrap_supports_poisson_counts():
    rng = np.random.default_rng(14)
    model = tomo.make_measurement_model(5, 2, 3, rng)
    basis = tomo.make_admissible_basis(3, rng, n_vectors=2)
    k_rel = tomo._normalize(tomo.project_admissible(rng.normal(size=3), basis))
    theta_hat = 0.05 * k_rel
    estimates, shrunk_cov, alpha = tomo.parametric_bootstrap_lambda(
        theta_hat,
        model,
        k_rel,
        basis,
        shots_per_setting=80,
        visibility=0.9,
        crosstalk=0.02,
        drift_level=0.01,
        count_model="poisson",
        reconstruction="linear",
        ridge=1e-3,
        n_boot=4,
        rng=rng,
    )
    assert estimates.shape == (4,)
    assert shrunk_cov.shape == (3, 3)
    assert 0.0 <= alpha <= 1.0


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


def test_estimate_power_map_supports_poisson_grid():
    config = tomo.TomographyConfig(
        dim=3,
        n_settings=5,
        n_outcomes=2,
        n_sim=4,
        n_null=6,
        n_boot=3,
        seed=15,
        count_model="poisson",
    )
    df, metadata = tomo.estimate_power_map(
        config,
        lambda_values=[0.03],
        n_total_values=[300],
        visibility_values=[0.9],
        crosstalk_values=[0.01],
        drift_values=[0.01],
    )
    assert len(df) == 1
    assert df.iloc[0]["count_model"] == "poisson"
    assert metadata["admissible_dim"] >= 1


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
