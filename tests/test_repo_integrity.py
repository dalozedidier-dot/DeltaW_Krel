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
    assert "fetch(\"data/switch_robustness_landscape.json\")" in app
    for text in (ci, extended, makefile):
        assert "realistic_tomography_pipeline.py" in text
        assert "realistic_tomography_smoke" in text


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
