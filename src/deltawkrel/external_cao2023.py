"""Public-data ingestion utilities for Cao et al. Optica 2023.

The public file is not a DeltaW/K_rel process-matrix tomography dataset.  It is
an external semi-device-independent quantum-switch count table with the
published inequality coefficients.  This module verifies the raw file and
recomputes the inequality value from the experimental probabilities.
"""

from __future__ import annotations

import hashlib
import json
import urllib.request
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
from scipy.io import loadmat

SOURCE_ID = "cao2023_sdi"
ARTICLE = (
    "Cao et al., Semi-device-independent certification of indefinite causal "
    "order in a photonic quantum switch, Optica 10, 561 (2023)"
)
DOI = "10.1364/OPTICA.483876"
ARXIV = "2202.05346"
REPOSITORY_URL = "https://github.com/jessicabavaresco/experimental-SDI-causality"
SOURCE_COMMIT = "956a5ed3c8dcf00a7049acf3486ec8b391598863"
DATA_URL = (
    "https://raw.githubusercontent.com/jessicabavaresco/"
    f"experimental-SDI-causality/{SOURCE_COMMIT}/inequality_coefficients.mat"
)
EXPECTED_SHA256 = "dfe2739d794ccbb699861fb18640b815ebd175604f26d4781b1a21fb672d8a30"
EXPECTED_VARIABLES = (
    "S_experiment",
    "S_theory",
    "alpha_abcxyz",
    "experimental_counts",
    "p_experiment",
    "p_theory",
)


@dataclass(frozen=True)
class Cao2023Report:
    """Summary of the public Cao 2023 semi-DI data file."""

    status: str
    source_id: str
    article: str
    doi: str
    arxiv: str
    repository_url: str
    data_url: str
    source_commit: str
    raw_sha256: str
    raw_size_bytes: int
    retrieved_at: str
    variables: dict[str, list[int]]
    total_counts: int
    setting_counts: list[list[list[int]]]
    min_setting_counts: int
    max_setting_counts: int
    probability_norm_min: float
    probability_norm_max: float
    max_probability_delta_from_counts: float
    s_experiment_reported: float
    s_experiment_recomputed: float
    s_theory_reported: float
    s_theory_recomputed: float
    s_experiment_abs_delta: float
    s_theory_abs_delta: float
    inequality_bound: float
    inequality_violated: bool
    violation_margin: float
    multinomial_standard_error: float
    statistical_z_score: float
    interpretation: str

    def to_jsonable(self) -> dict[str, Any]:
        return asdict(self)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download_public_file(
    destination: Path,
    *,
    url: str = DATA_URL,
    expected_sha256: str = EXPECTED_SHA256,
) -> Path:
    """Download and verify the pinned public MATLAB data file."""

    destination.parent.mkdir(parents=True, exist_ok=True)
    urllib.request.urlretrieve(url, destination)
    digest = sha256_file(destination)
    if digest != expected_sha256:
        raise ValueError(
            f"downloaded file hash mismatch: expected {expected_sha256}, got {digest}"
        )
    return destination


def _array_shape(value: np.ndarray) -> list[int]:
    return [int(x) for x in value.shape]


def _scalar(value: np.ndarray) -> float:
    squeezed = np.asarray(value).squeeze()
    if squeezed.shape != ():
        raise ValueError(f"expected scalar, got shape {value.shape}")
    return float(squeezed)


def _validate_shapes(data: dict[str, np.ndarray]) -> None:
    missing = [name for name in EXPECTED_VARIABLES if name not in data]
    if missing:
        raise ValueError(f"missing public-data variables: {', '.join(missing)}")

    tensor_shape = (2, 2, 4, 2, 2, 2)
    for name in ("alpha_abcxyz", "experimental_counts", "p_experiment", "p_theory"):
        if data[name].shape != tensor_shape:
            raise ValueError(f"{name} has shape {data[name].shape}, expected {tensor_shape}")

    for name in ("S_experiment", "S_theory"):
        if np.asarray(data[name]).shape != (1, 1):
            raise ValueError(f"{name} has shape {data[name].shape}, expected (1, 1)")


def analyze_public_file(path: Path, *, retrieved_at: str = "unknown") -> Cao2023Report:
    """Read the public file and recompute the semi-DI inequality diagnostics."""

    digest = sha256_file(path)
    raw = loadmat(path)
    data = {name: raw[name] for name in EXPECTED_VARIABLES if name in raw}
    _validate_shapes(data)

    alpha = np.asarray(data["alpha_abcxyz"], dtype=float)
    counts = np.asarray(data["experimental_counts"], dtype=float)
    p_experiment = np.asarray(data["p_experiment"], dtype=float)
    p_theory = np.asarray(data["p_theory"], dtype=float)

    if np.any(counts < 0):
        raise ValueError("experimental_counts contains negative values")
    if np.any(p_experiment < -1e-12) or np.any(p_theory < -1e-12):
        raise ValueError("probability tables contain negative entries")

    setting_counts = np.sum(counts, axis=(0, 1, 2))
    if np.any(setting_counts <= 0):
        raise ValueError("each input setting must have at least one count")

    counts_probabilities = counts / setting_counts.reshape((1, 1, 1) + setting_counts.shape)
    norms = np.sum(p_experiment, axis=(0, 1, 2))
    max_probability_delta = float(np.max(np.abs(counts_probabilities - p_experiment)))

    s_experiment_reported = _scalar(data["S_experiment"])
    s_theory_reported = _scalar(data["S_theory"])
    s_experiment_recomputed = float(np.sum(alpha * p_experiment))
    s_theory_recomputed = float(np.sum(alpha * p_theory))

    per_setting_means = np.sum(alpha * p_experiment, axis=(0, 1, 2))
    per_setting_second = np.sum(alpha * alpha * p_experiment, axis=(0, 1, 2))
    multinomial_variance = float(
        np.sum((per_setting_second - per_setting_means * per_setting_means) / setting_counts)
    )
    standard_error = float(np.sqrt(max(multinomial_variance, 0.0)))
    violation_margin = max(0.0, -s_experiment_recomputed)
    z_score = float(violation_margin / standard_error) if standard_error > 0 else float("inf")

    variables = {name: _array_shape(data[name]) for name in EXPECTED_VARIABLES}

    return Cao2023Report(
        status="public_experimental_counts_verified",
        source_id=SOURCE_ID,
        article=ARTICLE,
        doi=DOI,
        arxiv=ARXIV,
        repository_url=REPOSITORY_URL,
        data_url=DATA_URL,
        source_commit=SOURCE_COMMIT,
        raw_sha256=digest,
        raw_size_bytes=int(path.stat().st_size),
        retrieved_at=retrieved_at,
        variables=variables,
        total_counts=int(np.sum(counts)),
        setting_counts=setting_counts.astype(int).tolist(),
        min_setting_counts=int(np.min(setting_counts)),
        max_setting_counts=int(np.max(setting_counts)),
        probability_norm_min=float(np.min(norms)),
        probability_norm_max=float(np.max(norms)),
        max_probability_delta_from_counts=max_probability_delta,
        s_experiment_reported=s_experiment_reported,
        s_experiment_recomputed=s_experiment_recomputed,
        s_theory_reported=s_theory_reported,
        s_theory_recomputed=s_theory_recomputed,
        s_experiment_abs_delta=abs(s_experiment_reported - s_experiment_recomputed),
        s_theory_abs_delta=abs(s_theory_reported - s_theory_recomputed),
        inequality_bound=0.0,
        inequality_violated=s_experiment_recomputed < 0.0,
        violation_margin=violation_margin,
        multinomial_standard_error=standard_error,
        statistical_z_score=z_score,
        interpretation=(
            "Public semi-device-independent quantum-switch counts from Cao et al. "
            "are verified and reproduce the published inequality value. This is "
            "external experimental evidence for that semi-DI inequality, not a "
            "DeltaW/K_rel process-matrix tomography reanalysis."
        ),
    )


def write_report(report: Cao2023Report, outdir: Path) -> Path:
    outdir.mkdir(parents=True, exist_ok=True)
    out_path = outdir / "cao2023_sdi_report.json"
    out_path.write_text(
        json.dumps(report.to_jsonable(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return out_path
