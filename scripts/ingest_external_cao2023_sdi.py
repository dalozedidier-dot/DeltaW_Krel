#!/usr/bin/env python
"""Verify the public Cao et al. 2023 semi-DI quantum-switch data file."""

from __future__ import annotations

import argparse
from datetime import UTC, datetime
from pathlib import Path

from deltawkrel.external_cao2023 import (
    DATA_URL,
    EXPECTED_SHA256,
    analyze_public_file,
    download_public_file,
    write_report,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        default=None,
        help="Existing inequality_coefficients.mat file. If omitted, download the pinned public file.",
    )
    parser.add_argument(
        "--raw-dir",
        type=Path,
        default=Path("results/external/cao2023_sdi/raw"),
        help="Directory used for the downloaded raw public file.",
    )
    parser.add_argument(
        "--outdir",
        type=Path,
        default=Path("results/external/cao2023_sdi"),
        help="Directory receiving cao2023_sdi_report.json.",
    )
    parser.add_argument(
        "--retrieved-at",
        default=datetime.now(UTC).date().isoformat(),
        help="Retrieval date written into the report.",
    )
    args = parser.parse_args()

    if args.input is None:
        raw_path = args.raw_dir / "inequality_coefficients.mat"
        print(f"Downloading public data: {DATA_URL}")
        print(f"Expected SHA-256: {EXPECTED_SHA256}")
        download_public_file(raw_path)
    else:
        raw_path = args.input

    report = analyze_public_file(raw_path, retrieved_at=args.retrieved_at)
    out_path = write_report(report, args.outdir)
    print(f"verified {report.total_counts} public experimental counts")
    print(f"S_experiment = {report.s_experiment_recomputed:.12f}")
    print(f"multinomial-only z = {report.statistical_z_score:.2f}")
    print(f"wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
