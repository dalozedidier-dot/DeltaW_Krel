"""Tests for the reproducibility pipeline scripts.

Covers the micro-tomography stress-test, the manifest validator, the
preregistration freeze, the reproducibility report generator and the SDP
validation runner, each on synthetic fixtures in temporary directories.
"""
from __future__ import annotations

import hashlib
import json
from types import SimpleNamespace
from pathlib import Path

import numpy as np
import pytest

import freeze_preregistration
import generate_manifest
import generate_reproducibility_report
import micro_tomography_simulation as micro
import run_certified_bounds as certified_bounds_cli
import run_certified_witness_analysis as certified_analysis
import run_certified_witness_landscape as certified_landscape
import run_finite_count_analysis as finite_count_cli
import validate_manifest
from deltawkrel.certified_bounds import CertifiedInterval, SolverResult


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


def test_sha256_file_normalizes_crlf(tmp_path):
    lf = tmp_path / "lf.txt"
    crlf = tmp_path / "crlf.txt"
    lf.write_bytes(b"a\nb\n")
    crlf.write_bytes(b"a\r\nb\r\n")
    assert validate_manifest.sha256_file(lf) == validate_manifest.sha256_file(crlf)
    assert generate_manifest.sha256_file(lf) == generate_manifest.sha256_file(crlf)


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

    git = generate_manifest.git_executable()
    subprocess.run([git, "init", "-q"], cwd=root, check=True)
    for rel_path, content in files.items():
        target = root / rel_path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")
    subprocess.run([git, "add", "-A"], cwd=root, check=True)


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
# certified witness CLI scripts
# ------------------------------------------------------------------


class _FakeWitness:
    R_g = 0.5
    S = np.eye(4)
    tightness_gap = 1e-12
    equality_residual = 1e-12
    status = "optimal"

    def value(self, process):
        return float(np.sum(self.S * process))


class _FakeCurve:
    lambdas = np.array([0.0, 0.5, 1.0])
    values = np.array([0.5, 0.2, -0.1])
    slope = -0.6
    intercept = 0.5
    affinity_residual = 1e-12
    zero_crossing = 0.8333333333333334

    def invert(self, observed_value):
        return (observed_value - self.intercept) / self.slope


def _toy_process(parameter=0.0):
    return (1.0 - float(parameter)) * np.eye(4)


def test_certified_witness_analysis_cli_with_fakes(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(certified_analysis, "ideal_quantum_switch_process", lambda: np.eye(4))
    monkeypatch.setattr(certified_analysis, "partially_dephased_switch_process", _toy_process)
    monkeypatch.setattr(certified_analysis, "dephased_switch_process", lambda: np.zeros((4, 4)))
    monkeypatch.setattr(certified_analysis, "switch_white_noise_process", lambda: 0.5 * np.eye(4))
    monkeypatch.setattr(certified_analysis, "switch_generalized_robustness_witness", lambda *a, **k: _FakeWitness())
    monkeypatch.setattr(certified_analysis, "affine_witness_curve", lambda *a, **k: _FakeCurve())
    monkeypatch.setattr(
        certified_analysis,
        "solve_switch_generalized_robustness",
        lambda *a, **k: SimpleNamespace(objective_value=1.0),
    )
    monkeypatch.setattr(
        certified_analysis,
        "admissible_direction",
        lambda *a, **k: SimpleNamespace(
            K_rel=np.eye(4),
            cos_angle_to_S=0.75,
            noise_leakage={"N0": 0.0},
            signal_response=1.0,
            retained_fraction=0.8,
        ),
    )
    certified_analysis.main()
    assert (tmp_path / "results" / "certified_witness_report.json").exists()
    assert (tmp_path / "results" / "certified_witness_curve.csv").exists()
    report = json.loads((tmp_path / "results" / "certified_witness_report.json").read_text(encoding="utf-8"))
    assert report["sdp_benchmark"]["R_g"] == 0.5


def test_certified_witness_landscape_cli_with_fakes(tmp_path, monkeypatch):
    monkeypatch.setattr(certified_landscape, "ideal_quantum_switch_process", lambda: np.eye(4))
    monkeypatch.setattr(certified_landscape, "switch_generalized_robustness_witness", lambda *a, **k: _FakeWitness())
    monkeypatch.setattr(
        certified_landscape,
        "solve_switch_generalized_robustness",
        lambda *a, **k: SimpleNamespace(objective_value=10.0),
    )
    monkeypatch.setattr(
        certified_landscape,
        "FAMILIES",
        {
            "control_dephasing": {
                "fn": lambda x: (1.0 - x) * np.eye(4),
                "reference": 0.0,
                "xlabel": "control dephasing",
                "verify_endpoints": [0.0, 1.0],
                "verify_full": [0.0, 0.5, 1.0],
            },
            "white_visibility": {
                "fn": lambda x: x * np.eye(4),
                "reference": 1.0,
                "xlabel": "visibility",
                "verify_endpoints": [1.0, 0.0],
                "verify_full": [1.0, 0.5, 0.0],
            },
            "order_bias": {
                "fn": lambda x: (1.0 - (x - 0.5) ** 2) * np.eye(4),
                "reference": 0.5,
                "xlabel": "order bias",
                "verify_endpoints": [0.0, 0.5, 1.0],
                "verify_full": [0.0, 0.5, 1.0],
            },
        },
    )
    monkeypatch.setattr(
        "sys.argv",
        ["run_certified_witness_landscape.py", "--outdir", str(tmp_path), "--dense-grid", "5"],
    )
    assert certified_landscape.main() == 0
    report = json.loads((tmp_path / "certified_witness_landscape.json").read_text(encoding="utf-8"))
    assert report["status"] == "certified_witness_landscape"
    assert set(report["families"]) == {"control_dephasing", "white_visibility", "order_bias"}


def test_certified_bounds_cli_with_fakes(tmp_path, monkeypatch):
    fake_interval = CertifiedInterval(
        R_g_lower=0.49,
        R_g_upper=0.51,
        width=0.02,
        solver_table=[
            SolverResult(
                solver="SCS",
                available=True,
                status="optimal",
                R_g_lower=0.49,
                R_g_upper=0.51,
                interval_width=0.02,
                equality_residual=1e-9,
                dual_subspace_residual=1e-9,
                min_eig_primal=-1e-10,
                note="fake",
            )
        ],
        consensus_spread=0.0,
    )
    monkeypatch.setattr(certified_bounds_cli, "ideal_quantum_switch_process", lambda: np.eye(4))
    monkeypatch.setattr(
        certified_bounds_cli,
        "certified_robustness_interval",
        lambda *a, **k: fake_interval,
    )
    monkeypatch.setattr(
        "sys.argv",
        ["run_certified_bounds.py", "--outdir", str(tmp_path), "--solvers", "SCS"],
    )
    assert certified_bounds_cli.main() == 0
    report = json.loads((tmp_path / "certified_bounds_report.json").read_text(encoding="utf-8"))
    assert report["status"] == "certified_bounds"
    assert report["R_g_interval"]["width"] == 0.02
    assert report["solver_table"][0]["solver"] == "SCS"


def test_finite_count_analysis_cli_with_fakes(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(finite_count_cli, "ideal_quantum_switch_process", lambda: np.eye(4))
    monkeypatch.setattr(finite_count_cli, "dephased_switch_process", lambda: np.zeros((4, 4)))
    monkeypatch.setattr(finite_count_cli, "switch_white_noise_process", lambda: 0.5 * np.eye(4))
    monkeypatch.setattr(finite_count_cli, "partially_dephased_switch_process", _toy_process)
    monkeypatch.setattr(
        finite_count_cli,
        "switch_generalized_robustness_witness",
        lambda *a, **k: SimpleNamespace(R_g=0.5, S=np.diag([1.0, 0.0, 0.0, 0.0])),
    )
    monkeypatch.setattr(
        finite_count_cli,
        "admissible_direction",
        lambda *a, **k: SimpleNamespace(K_rel=np.eye(4), cos_angle_to_S=0.5),
    )
    monkeypatch.setattr(
        finite_count_cli,
        "lambda_estimator_scaling",
        lambda *a, **k: SimpleNamespace(
            N_grid=np.array([100, 1000]),
            var_lambda_analytic=np.array([1e-3, 1e-4]),
            var_lambda_empirical=np.array([1.1e-3, 0.9e-4]),
            slope_rho=-0.2,
            per_copy_variance_ref=0.01,
        ),
    )
    monkeypatch.setattr(finite_count_cli, "copies_to_certify", lambda *a, **k: 5.0)
    monkeypatch.setattr(
        finite_count_cli,
        "false_positive_under_drift",
        lambda *a, **k: SimpleNamespace(
            drift_grid=np.array([0.0, 1.0]),
            fp_rate_krel=np.array([0.01, 0.01]),
            fp_rate_raw=np.array([0.0, 1.0]),
            krel_signal_shift=np.array([0.0, 0.0]),
            raw_signal_shift=np.array([-0.1, 0.1]),
        ),
    )
    finite_count_cli.main()
    report = json.loads((tmp_path / "results" / "finite_count_report.json").read_text(encoding="utf-8"))
    assert report["scalars_measured"] == 1
    assert report["one_over_N_scaling"]["scaling_confirmed"] is True
    assert (tmp_path / "results" / "finite_count.png").exists()


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
    realistic = results / "realistic_tomography_smoke"
    realistic.mkdir()
    (realistic / "realistic_tomography_report.json").write_text(
        json.dumps({"status": "realistic_tomography_simulation"}),
        encoding="utf-8",
    )
    full_tomo = results / "full_tomography_smoke"
    full_tomo.mkdir()
    (full_tomo / "full_tomography_report.json").write_text(
        json.dumps({"status": "full_tomography_simulation"}),
        encoding="utf-8",
    )
    external = results / "external" / "cao2023_sdi"
    external.mkdir(parents=True)
    (external / "cao2023_sdi_report.json").write_text(
        json.dumps({"status": "public_experimental_counts_verified"}),
        encoding="utf-8",
    )
    monkeypatch.setattr("sys.argv", ["generate_reproducibility_report.py"])
    assert generate_reproducibility_report.main() == 0
    report = (results / "reproducibility_report.md").read_text(encoding="utf-8")
    assert "scenario: toy_scenario" in report
    assert "config sha256: `abc123`" in report
    assert "white_noise: optimal" in report
    assert "Realistic tomography bridge output: present" in report
    assert "Full tomography stress output: present" in report
    assert "Cao 2023 public-count verification: present" in report
    assert "manifest entries: 1" in report


def test_report_marks_legacy_smoke_artifacts(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    legacy_mc = tmp_path / "monte_carlo_outputs_control"
    legacy_micro = tmp_path / "outputs"
    legacy_mc.mkdir()
    legacy_micro.mkdir()
    (legacy_mc / "monte_carlo_power_results.json").write_text("[]", encoding="utf-8")
    (legacy_micro / "micro_tomography_power.csv").write_text("power\n", encoding="utf-8")
    monkeypatch.setattr("sys.argv", ["generate_reproducibility_report.py"])
    assert generate_reproducibility_report.main() == 0
    report = (tmp_path / "results" / "reproducibility_report.md").read_text(encoding="utf-8")
    assert "Monte Carlo smoke output: legacy-present" in report
    assert "Micro-tomography smoke output: legacy-present" in report


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
