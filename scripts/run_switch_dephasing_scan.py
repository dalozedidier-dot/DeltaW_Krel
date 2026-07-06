#!/usr/bin/env python
"""Scan generalized robustness along partial control dephasing of the switch.

The scan maps the falsification curve

    W(lambda) = (1 - lambda) W_switch + lambda W_dephased,

where lambda=0 is the ideal quantum switch and lambda=1 is the fully dephased
causally separable negative control.  Results are exported as strict JSON and
CSV so they can be used directly in a supplement figure.
"""
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
from deltawkrel.switch_models import partially_dephased_switch_process

DEFAULT_LAMBDAS = (0.0, 0.25, 0.50, 0.65, 0.70, 0.75, 1.0)


def parse_lambdas(raw: str) -> list[float]:
    values = [float(part.strip()) for part in raw.split(",") if part.strip()]
    if not values:
        raise ValueError("at least one lambda value is required.")
    bad = [v for v in values if not (0.0 <= v <= 1.0)]
    if bad:
        raise ValueError(f"lambda values must lie in [0, 1], got {bad}.")
    return values


def _json_safe(obj):
    if isinstance(obj, dict):
        return {k: _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(v) for v in obj]
    if isinstance(obj, float) and not math.isfinite(obj):
        return None
    return obj


def scan_dephasing_curve(
    lambdas: list[float],
    solver: str = "SCS",
    eps: float = 1e-8,
    max_iters: int = 200_000,
) -> list[dict]:
    rows = []
    for lam in lambdas:
        W = partially_dephased_switch_process(lam)
        diag = solve_switch_generalized_robustness(W, solver=solver, eps=eps, max_iters=max_iters)
        rows.append(
            {
                "lambda_dephasing": lam,
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


def summarize_curve(rows: list[dict], tolerance: float = 1e-6) -> dict:
    ordered = sorted(rows, key=lambda row: row["lambda_dephasing"])
    robust = [float(row["generalized_robustness"]) for row in ordered]
    monotone_nonincreasing = all(a + tolerance >= b for a, b in zip(robust, robust[1:]))
    positive_before_classical = all(
        row["generalized_robustness"] > tolerance
        for row in ordered
        if row["lambda_dephasing"] < 1.0 - tolerance
    )
    classical_endpoint_zero = any(
        abs(row["lambda_dephasing"] - 1.0) <= tolerance
        and abs(row["generalized_robustness"]) <= tolerance
        for row in ordered
    )
    return {
        "monotone_nonincreasing": monotone_nonincreasing,
        "positive_for_sampled_lambda_below_1": positive_before_classical,
        "classical_endpoint_zero": classical_endpoint_zero,
        "interpretation": (
            "On the sampled partial-dephasing family, generalized robustness "
            "decreases toward the fully dephased classical endpoint. Strict "
            "positivity below lambda=1 is an empirical SDP scan statement for "
            "the sampled grid, not an analytic proof for the continuum."
        ),
    }


def write_outputs(rows: list[dict], outdir: Path) -> tuple[Path, Path]:
    outdir.mkdir(parents=True, exist_ok=True)
    json_path = outdir / "switch_dephasing_scan.json"
    csv_path = outdir / "switch_dephasing_scan.csv"
    report = {
        "status": "switch_dephasing_scan",
        "family": "W(lambda) = (1 - lambda) W_switch + lambda W_dephased",
        "lambda_0": "ideal coherent switch",
        "lambda_1": "fully control-dephased causally separable switch",
        "rows": rows,
        "summary": summarize_curve(rows),
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
        "--lambdas",
        default=",".join(str(x) for x in DEFAULT_LAMBDAS),
        help="Comma-separated dephasing values in [0,1].",
    )
    parser.add_argument("--outdir", default="results", help="Output directory.")
    parser.add_argument("--solver", default="SCS", help="CVXPY solver name.")
    parser.add_argument("--eps", type=float, default=1e-8, help="SCS tolerance.")
    parser.add_argument("--max-iters", type=int, default=200_000, help="SCS iteration limit.")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    lambdas = parse_lambdas(args.lambdas)
    rows = scan_dephasing_curve(lambdas, solver=args.solver, eps=args.eps, max_iters=args.max_iters)
    json_path, csv_path = write_outputs(rows, Path(args.outdir))
    print(f"wrote {json_path}")
    print(f"wrote {csv_path}")
    for row in rows:
        print(
            f"lambda={row['lambda_dephasing']:.3f} "
            f"R_g={row['generalized_robustness']:.6f} "
            f"gap={row['witness_certificate_gap']:.2e}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
