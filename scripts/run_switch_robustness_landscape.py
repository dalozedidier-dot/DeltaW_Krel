#!/usr/bin/env python
"""Run multi-family SDP robustness scans around the ideal quantum switch."""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parents[1] / "src"
if _SRC.is_dir() and str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from deltawkrel.sdp import solve_switch_generalized_robustness
from deltawkrel.switch_models import (
    biased_coherent_switch_process,
    partially_dephased_switch_process,
    white_visibility_switch_process,
)

DEFAULT_GRIDS = {
    "control_dephasing": (0.0, 0.25, 0.50, 0.65, 0.70, 0.75, 1.0),
    "white_visibility": (0.0, 0.25, 0.50, 0.75, 1.0),
    "order_bias": (0.0, 0.10, 0.25, 0.50, 0.75, 0.90, 1.0),
}


def parse_grid(raw: str) -> list[float]:
    values = [float(part.strip()) for part in raw.split(",") if part.strip()]
    if not values:
        raise ValueError("at least one grid value is required.")
    bad = [v for v in values if not (0.0 <= v <= 1.0)]
    if bad:
        raise ValueError(f"grid values must lie in [0, 1], got {bad}.")
    return values


def _json_safe(obj):
    if isinstance(obj, dict):
        return {k: _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(v) for v in obj]
    if isinstance(obj, float) and not math.isfinite(obj):
        return None
    return obj


def process_for_family(family: str, value: float):
    if family == "control_dephasing":
        return partially_dephased_switch_process(value)
    if family == "white_visibility":
        return white_visibility_switch_process(value)
    if family == "order_bias":
        return biased_coherent_switch_process(value)
    raise ValueError(f"unknown family: {family}")


def scan_family(
    family: str,
    values: list[float],
    solver: str = "SCS",
    eps: float = 1e-8,
    max_iters: int = 200_000,
) -> list[dict]:
    rows = []
    for value in values:
        diag = solve_switch_generalized_robustness(
            process_for_family(family, value),
            solver=solver,
            eps=eps,
            max_iters=max_iters,
        )
        rows.append(
            {
                "family": family,
                "parameter": value,
                "generalized_robustness": diag.objective_value,
                "status": diag.status,
                "solver": diag.solver,
                "solver_version": diag.solver_version,
                "cvxpy_version": diag.cvxpy_version,
                "num_iters": diag.num_iters,
                "solve_time_s": diag.solve_time_s,
                "equality_residual_fro": diag.equality_residual_fro,
                "subspace_residual_fro": diag.subspace_residual_fro,
                "min_eig_W_AB": diag.min_eig_W_AB,
                "min_eig_W_BA": diag.min_eig_W_BA,
                "witness_value": diag.witness_value,
                "witness_certificate_gap": diag.witness_certificate_gap,
            }
        )
    return rows


def summarize_rows(rows: list[dict], tolerance: float = 1e-6) -> dict:
    by_family: dict[str, list[dict]] = {}
    for row in rows:
        by_family.setdefault(row["family"], []).append(row)
    summary = {}
    for family, family_rows in by_family.items():
        ordered = sorted(family_rows, key=lambda row: row["parameter"])
        values = [float(row["generalized_robustness"]) for row in ordered]
        summary[family] = {
            "min_robustness": min(values),
            "max_robustness": max(values),
            "all_status_optimal": all(row["status"] in {"optimal", "optimal_inaccurate"} for row in ordered),
            "max_witness_gap": max(abs(float(row["witness_certificate_gap"])) for row in ordered),
            "zero_parameters": [
                row["parameter"] for row in ordered if abs(row["generalized_robustness"]) <= tolerance
            ],
        }
        if family == "control_dephasing":
            summary[family]["monotone_nonincreasing"] = all(
                a + tolerance >= b for a, b in zip(values, values[1:])
            )
        if family == "white_visibility":
            summary[family]["monotone_nondecreasing"] = all(
                a <= b + tolerance for a, b in zip(values, values[1:])
            )
    return summary


def write_outputs(rows: list[dict], outdir: Path) -> tuple[Path, Path]:
    outdir.mkdir(parents=True, exist_ok=True)
    json_path = outdir / "switch_robustness_landscape.json"
    csv_path = outdir / "switch_robustness_landscape.csv"
    report = {
        "status": "switch_robustness_landscape",
        "families": {
            "control_dephasing": "lambda=0 ideal switch; lambda=1 fully dephased switch.",
            "white_visibility": "visibility=1 ideal switch; visibility=0 white-noise process.",
            "order_bias": "q=1/2 balanced switch; q=0 or q=1 fixed-order endpoints.",
        },
        "rows": rows,
        "summary": summarize_rows(rows),
    }
    json_path.write_text(
        json.dumps(_json_safe(report), indent=2, ensure_ascii=False, allow_nan=False),
        encoding="utf-8",
    )
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(_json_safe(rows))
    return json_path, csv_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--families",
        default="control_dephasing,white_visibility,order_bias",
        help="Comma-separated families to scan.",
    )
    parser.add_argument("--grid", default="", help="Override grid for all selected families.")
    parser.add_argument("--outdir", default="results", help="Output directory.")
    parser.add_argument("--solver", default="SCS", help="CVXPY solver name.")
    parser.add_argument("--eps", type=float, default=1e-8, help="SCS tolerance.")
    parser.add_argument("--max-iters", type=int, default=200_000, help="SCS iteration limit.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    families = [part.strip() for part in args.families.split(",") if part.strip()]
    rows = []
    for family in families:
        values = parse_grid(args.grid) if args.grid else list(DEFAULT_GRIDS[family])
        rows.extend(scan_family(family, values, solver=args.solver, eps=args.eps, max_iters=args.max_iters))
    json_path, csv_path = write_outputs(rows, Path(args.outdir))
    print(f"wrote {json_path}")
    print(f"wrote {csv_path}")
    for row in rows:
        print(
            f"{row['family']} p={row['parameter']:.3f} "
            f"R_g={row['generalized_robustness']:.6f} "
            f"gap={row['witness_certificate_gap']:.2e}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
