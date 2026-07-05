"""Tests for the reproducibility pipeline scripts.

Covers the micro-tomography stress-test, the manifest validator, the
preregistration freeze, the reproducibility report generator and the SDP
validation runner, each on synthetic fixtures in temporary directories.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

import freeze_preregistration
import generate_manifest
import generate_reproducibility_report
import micro_tomography_simulation as micro
import validate_manifest


# ------------------------------------------------------------------
# micro_tomography_simulation
# ------------------------------------------------------------------


def test_make_design_is_centered_and_normalized():
    rng = np.random.default_rng(0)
    k = micro.make_design(32, rng)
    assert abs(float(np.mean(k))) < 1e-12
    assert np.isclose(np.linalg.norm(k), 1.0)


def test_simulate_lr_shape_and_signal_ordering():
    rng = np.random.default_rng(1)
    k = micro.make_design(16, rng)
    lr_null = micro.simulate_lr(0.0, 0.97, 50_000, k, 300, rng)
    lr_alt = micro.simulate_lr(0.05, 0.97, 50_000, k, 300, rng)
    assert lr_null.shape == (300,)
    assert np.all(lr_null >= 0.0)
    assert float(np.mean(lr_alt)) > float(np.mean(lr_null))


def test_estimate_power_grid_and_type_one_error():
    df = micro.estimate_power(
        lambda_values=[0.0, 0.05],
        visibilities=[0.95, 0.99],
        n_values=[20_000],
        n_config=16,
        n_sim=200,
        n_null=400,
        seed=7,
    )
    assert len(df) == 1 * 2 * 2
    assert set(df.columns) >= {"visibility", "n_total", "lambda_true", "power", "lr_crit"}
    assert df["power"].between(0.0, 1.0).all()
    null_rows = df[df["lambda_true"] == 0.0]
    assert (null_rows["power"] <= 0.10).all()


def test_plot_power_writes_figure(tmp_path):
    df = micro.estimate_power(
        lambda_values=[0.0, 0.05],
        visibilities=[0.95, 0.99],
        n_values=[10_000],
        n_config=8,
        n_sim=50,
        n_null=100,
        seed=8,
    )
    out = tmp_path / "power.png"
    micro.plot_power(df, out)
    assert out.exists() and out.stat().st_size > 0


def test_micro_main_cli(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "sys.argv",
        [
            "micro_tomography_simulation.py",
            "--outdir", str(tmp_path),
            "--n-config", "8",
            "--n-sim", "20",
            "--n-null", "50",
            "--seed", "3",
        ],
    )
    micro.main()
    assert (tmp_path / "micro_tomography_power.csv").exists()
    assert (tmp_path / "power_micro_tomography.png").exists()


# ------------------------------------------------------------------
# validate_manifest
# ------------------------------------------------------------------


def _write_manifest(root: Path, mapping: dict[str, str]) -> Path:
    manifest = root / "MANIFEST.sha256.json"
    manifest.write_text(json.dumps(mapping), encoding="utf-8")
    return manifest


def test_sha256_file_matches_hashlib(tmp_path):
    payload = b"deltaw krel"
    target = tmp_path / "data.bin"
    target.write_bytes(payload)
    assert validate_manifest.sha256_file(target) == hashlib.sha256(payload).hexdigest()


def test_validate_manifest_success(tmp_path, monkeypatch, capsys):
    target = tmp_path / "a.txt"
    target.write_text("hello", encoding="utf-8")
    _write_manifest(tmp_path, {"a.txt": validate_manifest.sha256_file(target)})
    monkeypatch.setattr("sys.argv", ["validate_manifest.py", "--root", str(tmp_path)])
    assert validate_manifest.main() == 0
    assert "Validated 1 files" in capsys.readouterr().out


def test_validate_manifest_detects_missing_file(tmp_path, monkeypatch, capsys):
    _write_manifest(tmp_path, {"gone.txt": "0" * 64})
    monkeypatch.setattr("sys.argv", ["validate_manifest.py", "--root", str(tmp_path)])
    assert validate_manifest.main() == 1
    assert "Missing files" in capsys.readouterr().out


def test_validate_manifest_detects_hash_mismatch(tmp_path, monkeypatch, capsys):
    target = tmp_path / "a.txt"
    target.write_text("hello", encoding="utf-8")
    _write_manifest(tmp_path, {"a.txt": "0" * 64})
    monkeypatch.setattr("sys.argv", ["validate_manifest.py", "--root", str(tmp_path)])
    assert validate_manifest.main() == 1
    out = capsys.readouterr().out
    assert "Hash mismatches" in out
    assert "expected" in out


# ------------------------------------------------------------------
# generate_manifest
# ------------------------------------------------------------------


def _init_git_repo(root: Path, files: dict[str, str]) -> None:
    import subprocess

    subprocess.run(["git", "init", "-q"], cwd=root, check=True)
    for rel_path, content in files.items():
        target = root / rel_path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")
    subprocess.run(["git", "add", "-A"], cwd=root, check=True)


def test_generate_manifest_covers_tracked_files_and_excludes_volatile(tmp_path, monkeypatch):
    _init_git_repo(
        tmp_path,
        {
            "src/module.py": "x = 1\n",
            "results/report.md": "volatile\n",
            "outputs/data.csv": "volatile\n",
            "MANIFEST.sha256.json": "{}",
        },
    )
    monkeypatch.setattr("sys.argv", ["generate_manifest.py", "--root", str(tmp_path)])
    assert generate_manifest.main() == 0
    manifest = json.loads((tmp_path / "MANIFEST.sha256.json").read_text(encoding="utf-8"))
    assert "src/module.py" in manifest
    assert not any(p.startswith(("results/", "outputs/")) for p in manifest)
    assert "MANIFEST.sha256.json" not in manifest
    assert manifest["src/module.py"] == generate_manifest.sha256_file(tmp_path / "src/module.py")


def test_generated_manifest_passes_validation(tmp_path, monkeypatch):
    _init_git_repo(tmp_path, {"a.txt": "hello\n", "docs/b.md": "world\n"})
    monkeypatch.setattr("sys.argv", ["generate_manifest.py", "--root", str(tmp_path)])
    assert generate_manifest.main() == 0
    monkeypatch.setattr("sys.argv", ["validate_manifest.py", "--root", str(tmp_path)])
    assert validate_manifest.main() == 0


# ------------------------------------------------------------------
# freeze_preregistration
# ------------------------------------------------------------------


def test_freeze_preregistration_lock_content(tmp_path, monkeypatch):
    config = {
        "scenario": "toy_scenario",
        "trace_convention": "Tr(W)=d_O",
        "thresholds": {"alpha": 0.01, "lambda_sens": 0.005},
        "solver": {"primary": "CLARABEL"},
        "witness": {"selection_rule": "minimum_HS_norm_on_dual_face"},
    }
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(config), encoding="utf-8")
    out_path = tmp_path / "results" / "lock.json"
    monkeypatch.setattr(
        "sys.argv",
        ["freeze_preregistration.py", "--config", str(config_path), "--out", str(out_path)],
    )
    assert freeze_preregistration.main() == 0
    lock = json.loads(out_path.read_text(encoding="utf-8"))
    assert lock["status"] == "locked"
    assert lock["scenario"] == "toy_scenario"
    assert lock["thresholds"]["alpha"] == 0.01
    assert lock["config_sha256"] == freeze_preregistration.sha256_file(config_path)


def test_freeze_preregistration_is_deterministic(tmp_path, monkeypatch):
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps({"scenario": "s"}), encoding="utf-8")
    outputs = []
    for name in ("lock1.json", "lock2.json"):
        out_path = tmp_path / name
        monkeypatch.setattr(
            "sys.argv",
            ["freeze_preregistration.py", "--config", str(config_path), "--out", str(out_path)],
        )
        assert freeze_preregistration.main() == 0
        outputs.append(out_path.read_bytes())
    assert outputs[0] == outputs[1]


# ------------------------------------------------------------------
# generate_reproducibility_report
# ------------------------------------------------------------------


def test_report_with_missing_artifacts(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr("sys.argv", ["generate_reproducibility_report.py"])
    assert generate_reproducibility_report.main() == 0
    report = (tmp_path / "results" / "reproducibility_report.md").read_text(encoding="utf-8")
    assert "# DeltaW/K_rel reproducibility report" in report
    assert "status: missing" in report
    assert "manifest entries: 0" in report


def test_report_with_present_artifacts(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    results = tmp_path / "results"
    results.mkdir()
    (results / "preregistration_lock.json").write_text(
        json.dumps(
            {
                "status": "locked",
                "scenario": "toy_scenario",
                "trace_convention": "Tr(W)=d_O",
                "config_sha256": "abc123",
            }
        ),
        encoding="utf-8",
    )
    (results / "sdp_validation_report.json").write_text(
        json.dumps(
            {
                "status": "infrastructure_validation_only",
                "ideal_quantum_switch_benchmark": {"implemented": False, "submission_blocker": True},
                "targets": {
                    "white_noise": {"status": "optimal", "objective_value": 0.0, "solver": "CLARABEL"}
                },
            }
        ),
        encoding="utf-8",
    )
    (tmp_path / "MANIFEST.sha256.json").write_text(json.dumps({"a": "0" * 64}), encoding="utf-8")
    monkeypatch.setattr("sys.argv", ["generate_reproducibility_report.py"])
    assert generate_reproducibility_report.main() == 0
    report = (results / "reproducibility_report.md").read_text(encoding="utf-8")
    assert "scenario: toy_scenario" in report
    assert "config sha256: `abc123`" in report
    assert "white_noise: optimal" in report
    assert "manifest entries: 1" in report


# ------------------------------------------------------------------
# run_sdp_validation (requires cvxpy)
# ------------------------------------------------------------------


def test_run_sdp_validation_writes_report(tmp_path, monkeypatch):
    pytest.importorskip("cvxpy")
    import run_sdp_validation

    monkeypatch.chdir(tmp_path)
    run_sdp_validation.main()
    report_path = tmp_path / "results" / "sdp_validation_report.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["status"] == "sdp_validation_with_switch_benchmark"
    assert set(report["targets"]) == {
        "white_noise",
        "fixed_order_A_before_B",
        "fixed_order_B_before_A",
        "causally_separable_mixture_q037",
    }
    for diag in report["targets"].values():
        assert diag["status"] in {"optimal", "optimal_inaccurate"}
        assert diag["objective_value"] < 1e-5

    benchmark = report["ideal_quantum_switch_benchmark"]
    assert benchmark["implemented"] is True
    assert benchmark["benchmark_passed"] is True
    assert benchmark["submission_blocker"] is False
    assert benchmark["reference_generalized_robustness"] == pytest.approx(0.5454)
    assert benchmark["computed_generalized_robustness"] == pytest.approx(0.5454, abs=2e-3)
    assert abs(benchmark["dephased_switch_diagnostics"]["objective_value"]) < 1e-6
    gen_diag = benchmark["generalized_robustness_diagnostics"]
    assert gen_diag["witness_certificate_gap"] < 1e-6
    assert gen_diag["solver_version"]
    # Strict JSON: NaN fields must have been converted to null.
    assert "NaN" not in report_path.read_text(encoding="utf-8")
