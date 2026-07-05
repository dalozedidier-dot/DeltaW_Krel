"""Integrity tests for repository artifacts: config, manifest shape, notebooks.

These tests pin the structure of the preregistration config and the manifest
format so that accidental edits break CI instead of silently invalidating the
reproducibility chain.
"""
from __future__ import annotations

import json
import re
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
