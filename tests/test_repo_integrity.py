"""Integrity tests for repository artifacts: config, manifest shape, notebooks.

These tests pin the structure of the preregistration config and the manifest
format so that accidental edits break CI instead of silently invalidating the
reproducibility chain.
"""
from __future__ import annotations

import json
import re
import csv
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]


def test_preregistration_config_structure():
    config = json.loads(
        (REPO_ROOT / "config" / "config_preregistration.json").read_text(encoding="utf-8")
    )
    assert config["scenario"]
    assert config["trace_convention"] == "Tr(W)=d_O"
    thresholds = config["thresholds"]
    for key in ("alpha", "eps_R", "eps_num", "lambda_sens", "lambda_ref", "omega_white_alert"):
        assert key in thresholds, f"missing preregistered threshold: {key}"
    assert 0.0 < thresholds["alpha"] < 0.5
    assert thresholds["lambda_ref"] < thresholds["lambda_sens"]
    assert config["witness"]["post_hoc_redefinition_allowed"] is False
    assert config["solver"]["environment_locked"] is True


def test_preregistered_artifacts_exist():
    config = json.loads(
        (REPO_ROOT / "config" / "config_preregistration.json").read_text(encoding="utf-8")
    )
    for label, rel_path in config["reproducibility_artifacts"].items():
        assert (REPO_ROOT / rel_path).exists(), f"preregistered artifact missing: {label} -> {rel_path}"


def test_manifest_is_well_formed():
    manifest = json.loads((REPO_ROOT / "MANIFEST.sha256.json").read_text(encoding="utf-8"))
    assert isinstance(manifest, dict) and manifest
    hex_sha = re.compile(r"^[0-9a-f]{64}$")
    for rel_path, digest in manifest.items():
        assert isinstance(rel_path, str) and rel_path
        assert not rel_path.startswith("/"), f"manifest path must be relative: {rel_path}"
        assert hex_sha.match(digest), f"invalid sha256 for {rel_path}: {digest}"


def test_manifest_matches_repository_content(monkeypatch):
    """The committed manifest must validate against the checkout.

    If this fails after editing tracked files, regenerate the manifest with
    `make manifest-update` (python scripts/generate_manifest.py).
    """
    import validate_manifest

    monkeypatch.setattr("sys.argv", ["validate_manifest.py", "--root", str(REPO_ROOT)])
    assert validate_manifest.main() == 0, "manifest drift — run `make manifest-update`"


def test_notebooks_parse_as_nbformat_v4():
    nbformat = pytest.importorskip("nbformat")
    notebooks = sorted((REPO_ROOT / "notebooks").glob("*.ipynb"))
    assert notebooks, "no notebooks found"
    for path in notebooks:
        nb = nbformat.read(path, as_version=4)
        nbformat.validate(nb)
        assert nb.cells, f"notebook has no cells: {path.name}"


def test_citation_file_present_and_nonempty():
    citation = REPO_ROOT / "CITATION.cff"
    text = citation.read_text(encoding="utf-8")
    assert "title" in text.lower()


def test_github_pages_site_is_present():
    index = REPO_ROOT / "site" / "index.html"
    css = REPO_ROOT / "site" / "styles.css"
    app = REPO_ROOT / "site" / "app.js"
    workflow = REPO_ROOT / ".github" / "workflows" / "pages.yml"
    assert index.exists(), "GitHub Pages index missing"
    assert css.exists(), "GitHub Pages stylesheet missing"
    assert app.exists(), "GitHub Pages JavaScript missing"
    assert workflow.exists(), "GitHub Pages workflow missing"
    html = index.read_text(encoding="utf-8")
    assert "R<sub>g</sub> = 0.545351" in html
    assert "run_switch_robustness_landscape.py" in html
    assert "data/switch_robustness_landscape.json" in html
    assert "data/switch_robustness_landscape.csv" in html
    assert "Advanced numerical validation" in html
    assert "data/full_tomography/full_tomography_report.json" in html
    assert "data/full_tomography/full_tomography_power.csv" in html
    assert "data/full_tomography/full_tomography_power.png" in html
    assert "Certified witness layer" in html
    assert "data/certified_witness/certified_witness_landscape.json" in html
    assert "data/certified_witness/certified_witness_landscape.csv" in html
    assert "data/certified_witness/certified_witness_landscape.png" in html
    assert "Ultimate roadmap" in html
    assert "docs/ULTIMATE_VISION_ROADMAP.md" in html
    assert "Control dephasing" in html
    assert "White visibility" in html
    assert "Order bias" in html


def test_github_pages_and_ci_reference_realistic_tomography_bridge():
    html = (REPO_ROOT / "site" / "index.html").read_text(encoding="utf-8")
    app = (REPO_ROOT / "site" / "app.js").read_text(encoding="utf-8")
    ci = (REPO_ROOT / ".github" / "workflows" / "ci.yml").read_text(encoding="utf-8")
    extended = (REPO_ROOT / ".github" / "workflows" / "tests-extended.yml").read_text(encoding="utf-8")
    makefile = (REPO_ROOT / "Makefile").read_text(encoding="utf-8")

    assert "realistic_tomography_pipeline.py" in html
    assert "full_realistic_tomography.py" in html
    assert "run_certified_witness_landscape.py" in html
    assert "fetch(\"data/switch_robustness_landscape.json\")" in app
    assert "fetch(\"data/certified_witness/certified_witness_landscape.json\")" in app
    assert "fetch(\"data/full_tomography/full_tomography_report.json\")" in app
    for text in (ci, extended, makefile):
        assert "realistic_tomography_pipeline.py" in text
        assert "realistic_tomography_smoke" in text
        assert "full_realistic_tomography.py" in text
        assert "full_tomography_smoke" in text


def test_github_pages_landscape_data_is_complete_and_strict():
    json_path = REPO_ROOT / "site" / "data" / "switch_robustness_landscape.json"
    csv_path = REPO_ROOT / "site" / "data" / "switch_robustness_landscape.csv"
    assert json_path.exists(), "published landscape JSON missing"
    assert csv_path.exists(), "published landscape CSV missing"
    text = json_path.read_text(encoding="utf-8")
    assert "NaN" not in text
    report = json.loads(text)
    assert report["status"] == "switch_robustness_landscape"
    rows = report["rows"]
    assert len(rows) == 19
    families = {row["family"] for row in rows}
    assert families == {"control_dephasing", "white_visibility", "order_bias"}
    lookup = {(row["family"], row["parameter"]): row for row in rows}
    assert lookup[("control_dephasing", 0.0)]["generalized_robustness"] == pytest.approx(0.545351, abs=5e-4)
    assert abs(lookup[("control_dephasing", 1.0)]["generalized_robustness"]) < 1e-6
    assert lookup[("white_visibility", 1.0)]["generalized_robustness"] == pytest.approx(0.545351, abs=5e-4)
    assert abs(lookup[("order_bias", 0.0)]["generalized_robustness"]) < 1e-6
    assert abs(lookup[("order_bias", 1.0)]["generalized_robustness"]) < 1e-6
    assert max(abs(row["witness_certificate_gap"]) for row in rows) < 1e-7

    with csv_path.open(newline="", encoding="utf-8") as handle:
        csv_rows = list(csv.DictReader(handle))
    assert len(csv_rows) == len(rows)


def test_github_pages_certified_witness_data_is_complete_and_strict():
    json_path = REPO_ROOT / "site" / "data" / "certified_witness" / "certified_witness_landscape.json"
    csv_path = REPO_ROOT / "site" / "data" / "certified_witness" / "certified_witness_landscape.csv"
    png_path = REPO_ROOT / "site" / "data" / "certified_witness" / "certified_witness_landscape.png"
    assert json_path.exists(), "published certified witness JSON missing"
    assert csv_path.exists(), "published certified witness CSV missing"
    assert png_path.exists(), "published certified witness PNG missing"
    assert png_path.stat().st_size > 10_000
    text = json_path.read_text(encoding="utf-8")
    assert "NaN" not in text
    report = json.loads(text)
    families = report["families"]
    assert set(families) == {"control_dephasing", "white_visibility", "order_bias"}
    assert report["reference_witness"]["R_g"] == pytest.approx(0.545351, abs=5e-4)
    assert families["control_dephasing"]["is_affine"] is True
    assert families["white_visibility"]["is_affine"] is True
    assert families["order_bias"]["is_affine"] is False
    for family in families.values():
        assert family["tight_at_reference"] is True
        assert family["lower_bound_holds_on_grid"] is True
        lo, hi = family["certified_nonseparable_region"]
        assert 0.0 <= lo < hi <= 1.0
        assert family["verification_grid"]

    with csv_path.open(newline="", encoding="utf-8") as handle:
        csv_rows = list(csv.DictReader(handle))
    assert len(csv_rows) == sum(len(family["verification_grid"]) for family in families.values())


def test_github_pages_full_tomography_data_is_complete_and_strict():
    json_path = REPO_ROOT / "site" / "data" / "full_tomography" / "full_tomography_report.json"
    csv_path = REPO_ROOT / "site" / "data" / "full_tomography" / "full_tomography_power.csv"
    png_path = REPO_ROOT / "site" / "data" / "full_tomography" / "full_tomography_power.png"
    assert json_path.exists(), "published full tomography JSON missing"
    assert csv_path.exists(), "published full tomography CSV missing"
    assert png_path.exists(), "published full tomography PNG missing"
    assert png_path.stat().st_size > 10_000
    text = json_path.read_text(encoding="utf-8")
    assert "NaN" not in text
    report = json.loads(text)
    assert report["status"] == "full_tomography_simulation"
    rows = report["rows"]
    assert len(rows) == 18
    assert {row["lambda_true"] for row in rows} == {0.0, 0.02, 0.05}
    assert {row["n_total"] for row in rows} == {800, 2000}
    assert {row["visibility"] for row in rows} == {0.9, 0.95, 0.97}
    assert {row["dephasing"] for row in rows} == {0.01}
    assert {row["control_crosstalk"] for row in rows} == {0.01}
    assert {row["operation_crosstalk"] for row in rows} == {0.01}
    assert {row["drift"] for row in rows} == {0.005}
    assert {row["path_loss"] for row in rows} == {0.02}
    for row in rows:
        assert 0.0 <= row["power"] <= 1.0
        assert 0.0 <= row["false_positive_miscalibrated"] <= 1.0
        assert row["applicability_passed"] is True
        assert row["sigma_lambda_shrunk"] > 0.0
        assert row["projected_direction_norm"] > 0.0
    assert max(row["false_positive_miscalibrated"] for row in rows) <= 0.05

    with csv_path.open(newline="", encoding="utf-8") as handle:
        csv_rows = list(csv.DictReader(handle))
    assert len(csv_rows) == len(rows)


def test_ultimate_vision_roadmap_is_present_but_not_overclaimed():
    roadmap = REPO_ROOT / "docs" / "ULTIMATE_VISION_ROADMAP.md"
    assert roadmap.exists(), "ultimate vision roadmap missing"
    text = roadmap.read_text(encoding="utf-8")
    assert "North star" in text
    assert "World-class numerical simulation" in text
    assert "Theoretical expansion" in text
    assert "Existing data and new experiments" in text
    assert "Non-negotiable guardrails" in text
    assert "Do not describe toy or smoke simulations as experimental evidence." in text
    assert "ambition tracker" in text


def test_docs_do_not_revert_switch_benchmark_status():
    checked = [
        REPO_ROOT / "docs" / "CLAIM_EVIDENCE_MATRIX.md",
        REPO_ROOT / "docs" / "SDP_VALIDATION_STATUS.md",
        REPO_ROOT / "docs" / "MAXIMIZE_REPOSITORY_FOR_SUBMISSION.md",
        REPO_ROOT / "docs" / "INSTALL_AND_TEST_REPORT.md",
        REPO_ROOT / "src" / "deltawkrel" / "__init__.py",
        REPO_ROOT / "src" / "deltawkrel" / "sdp.py",
    ]
    forbidden = (
        "ideal quantum switch process is **not** implemented",
        "ideal quantum-switch benchmark remains intentionally marked as not implemented",
        "ideal_quantum_switch_process()` raises `NotImplementedError`",
        "not completed quantum-switch benchmark",
    )
    for path in checked:
        text = path.read_text(encoding="utf-8").lower()
        for phrase in forbidden:
            assert phrase.lower() not in text, f"stale switch status in {path}"
