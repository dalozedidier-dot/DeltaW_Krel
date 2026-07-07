from __future__ import annotations

import numpy as np
import pytest
from scipy.io import savemat

from deltawkrel.external_cao2023 import analyze_public_file, download_public_file


def _write_fixture(path, *, include_all: bool = True) -> None:
    counts = np.ones((2, 2, 4, 2, 2, 2), dtype=float) * 10
    p_experiment = counts / counts.sum(axis=(0, 1, 2), keepdims=True)
    p_theory = p_experiment.copy()
    alpha = np.zeros_like(p_experiment)
    alpha[0, 0, 0, :, :, :] = -1.0
    s_value = float(np.sum(alpha * p_experiment))
    payload = {
        "S_experiment": np.array([[s_value]]),
        "S_theory": np.array([[s_value]]),
        "alpha_abcxyz": alpha,
        "experimental_counts": counts,
        "p_experiment": p_experiment,
        "p_theory": p_theory,
    }
    if not include_all:
        payload.pop("p_theory")
    savemat(path, payload)


def test_analyze_public_file_recomputes_sdi_inequality(tmp_path):
    mat_path = tmp_path / "cao_fixture.mat"
    _write_fixture(mat_path)

    report = analyze_public_file(mat_path, retrieved_at="test")

    assert report.status == "public_experimental_counts_verified"
    assert report.total_counts == 2 * 2 * 4 * 2 * 2 * 2 * 10
    assert report.variables["experimental_counts"] == [2, 2, 4, 2, 2, 2]
    assert report.s_experiment_recomputed == pytest.approx(report.s_experiment_reported)
    assert report.s_theory_recomputed == pytest.approx(report.s_theory_reported)
    assert report.s_experiment_abs_delta < 1e-12
    assert report.max_probability_delta_from_counts == pytest.approx(0.0)
    assert report.inequality_violated is True
    assert report.violation_margin > 0.0
    assert report.multinomial_standard_error > 0.0


def test_analyze_public_file_rejects_missing_variables(tmp_path):
    mat_path = tmp_path / "cao_missing.mat"
    _write_fixture(mat_path, include_all=False)

    with pytest.raises(ValueError, match="missing public-data variables"):
        analyze_public_file(mat_path)


def test_download_public_file_rejects_hash_mismatch(tmp_path, monkeypatch):
    source = tmp_path / "source.mat"
    source.write_bytes(b"not the public file")

    def fake_urlretrieve(url, destination):
        del url
        destination.write_bytes(source.read_bytes())

    monkeypatch.setattr("urllib.request.urlretrieve", fake_urlretrieve)

    with pytest.raises(ValueError, match="hash mismatch"):
        download_public_file(tmp_path / "downloaded.mat", expected_sha256="0" * 64)
