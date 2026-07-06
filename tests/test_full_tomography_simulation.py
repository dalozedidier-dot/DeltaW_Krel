from __future__ import annotations

import json

import numpy as np
import pytest

import full_realistic_tomography as full_alias
import full_tomography_simulation as full
import realistic_tomography_pipeline as tomo


def test_full_realistic_tomography_wrapper_reuses_full_pipeline_entrypoint():
    assert full_alias.main is full.main


def test_effective_noise_parameters_and_path_loss_validate_bounds():
    assert full.effective_visibility(0.9, 0.1, 0.2) == pytest.approx(0.648)
    assert full.effective_crosstalk(0.1, 0.2) == pytest.approx(0.28)
    shots = full.path_loss_shots(120, 6, 0.2)
    assert shots.shape == (6,)
    assert np.all(shots >= 1)
    assert np.min(shots) < np.max(shots)
    with pytest.raises(ValueError):
        full.effective_visibility(1.2, 0.0, 0.0)
    with pytest.raises(ValueError):
        full.path_loss_shots(100, 4, 1.0)


def test_variable_counts_and_interleaved_baseline_reconstruction():
    rng = np.random.default_rng(1)
    model = tomo.make_measurement_model(8, 2, 4, rng)
    basis = tomo.make_admissible_basis(4, rng, n_vectors=2)
    k_rel = tomo._normalize(tomo.project_admissible(rng.normal(size=4), basis))
    theta = 0.15 * k_rel
    probs = tomo.probabilities_from_theta(model, theta, visibility=0.9, drift_level=0.02)
    shots = full.path_loss_shots(4000, 8, 0.1)
    counts = full.generate_variable_counts(probs, shots, rng, "multinomial")
    assert counts.shape == probs.shape
    assert np.all(counts.sum(axis=1) == shots)

    baseline = tomo.probabilities_from_theta(model, np.zeros(4), visibility=0.9, drift_level=0.02)
    baseline_counts = full.generate_variable_counts(baseline, shots, rng, "multinomial")
    theta_hat = full.reconstruct_linear_with_baseline(
        counts,
        tomo.counts_to_frequencies(baseline_counts),
        model,
        ridge=1e-4,
        visibility_assumed=0.9,
    )
    assert theta_hat.shape == (4,)
    assert np.isfinite(full.tomo.lambda_hat(theta_hat, k_rel, basis))


def test_simulate_estimate_supports_none_and_interleaved_correction():
    rng = np.random.default_rng(2)
    model = tomo.make_measurement_model(6, 2, 3, rng)
    basis = tomo.make_admissible_basis(3, rng, n_vectors=2)
    k_rel = tomo._normalize(tomo.project_admissible(rng.normal(size=3), basis))
    shots = full.path_loss_shots(600, 6, 0.05)
    for correction in ("none", "interleaved"):
        estimate, theta_hat, probs = full.simulate_estimate(
            model,
            k_rel,
            basis,
            lambda_true=0.04,
            shots_by_setting=shots,
            visibility=0.95,
            dephasing=0.02,
            control_crosstalk=0.01,
            operation_crosstalk=0.01,
            drift=0.005,
            correction=correction,
            count_model="multinomial",
            reconstruction="linear",
            ridge=1e-3,
            rng=rng,
        )
        assert np.isfinite(estimate)
        assert theta_hat.shape == (3,)
        assert np.allclose(probs.sum(axis=1), 1.0)
    with pytest.raises(ValueError):
        full.simulate_estimate(
            model,
            k_rel,
            basis,
            lambda_true=0.04,
            shots_by_setting=shots,
            visibility=0.95,
            dephasing=0.02,
            control_crosstalk=0.01,
            operation_crosstalk=0.01,
            drift=0.005,
            correction="interleaved",
            count_model="multinomial",
            reconstruction="mle",
            ridge=1e-3,
            rng=rng,
        )


def test_likelihood_ratio_and_applicability_record():
    lr = full.likelihood_ratio(np.array([-1.0, 0.0, 2.0]), null_sd=2.0)
    assert np.allclose(lr, [0.0, 0.0, 1.0])
    config = full.FullTomographyConfig(max_miscalibrated_fpr=0.05)
    ok = full.applicability_record(
        projected_direction_norm=1.0,
        null_sd=0.1,
        lambda_true=0.05,
        visibility_eff=0.8,
        false_positive_miscalibrated=0.01,
        config=config,
    )
    bad = full.applicability_record(
        projected_direction_norm=0.0,
        null_sd=0.1,
        lambda_true=0.05,
        visibility_eff=0.8,
        false_positive_miscalibrated=0.2,
        config=config,
    )
    assert ok["applicability_passed"] is True
    assert bad["applicability_passed"] is False


def test_estimate_full_map_and_outputs_are_strict(tmp_path):
    config = full.FullTomographyConfig(
        dim=3,
        n_settings=5,
        n_outcomes=2,
        n_sim=4,
        n_null=6,
        n_boot=3,
        seed=3,
        correction="interleaved",
    )
    df, metadata = full.estimate_full_map(
        config,
        lambda_values=[0.0, 0.05],
        n_total_values=[500],
        visibility_values=[0.95],
        dephasing_values=[0.01],
        control_crosstalk_values=[0.01],
        operation_crosstalk_values=[0.01],
        drift_values=[0.005],
        path_loss_values=[0.02],
    )
    assert len(df) == 2
    assert set(df.columns) >= {
        "power",
        "lr_crit",
        "false_positive_miscalibrated",
        "applicability_passed",
        "effective_visibility",
        "effective_crosstalk",
    }
    assert metadata["status"] == "full_tomography_simulation"
    csv_path, json_path, png_path = full.write_outputs(df, metadata, tmp_path)
    assert csv_path.exists()
    assert png_path.exists()
    text = json_path.read_text(encoding="utf-8")
    assert "NaN" not in text
    report = json.loads(text)
    assert report["status"] == "full_tomography_simulation"
    assert len(report["rows"]) == 2


def test_full_tomography_cli_smoke(tmp_path):
    code = full.main(
        [
            "--outdir",
            str(tmp_path),
            "--dim",
            "3",
            "--n-settings",
            "5",
            "--lambda-values",
            "0,0.04",
            "--n-total-values",
            "500",
            "--visibility-values",
            "0.95",
            "--dephasing-values",
            "0.01",
            "--control-crosstalk-values",
            "0.01",
            "--operation-crosstalk-values",
            "0.01",
            "--drift-values",
            "0.005",
            "--path-loss-values",
            "0.02",
            "--n-sim",
            "3",
            "--n-null",
            "5",
            "--n-boot",
            "3",
        ]
    )
    assert code == 0
    assert (tmp_path / "full_tomography_report.json").exists()
