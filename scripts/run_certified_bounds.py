#!/usr/bin/env python
r"""A5 - certified numerical interval for switch generalized robustness.

This script turns the SDP benchmark into a reviewer-facing certificate report:
the primal value gives an upper bound, the dual witness value gives a lower
bound, and the remaining width is the numerical certificate gap.  The solver
table keeps unavailable or failing solvers explicit instead of silently hiding
them.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import asdict
from pathlib import Path

import numpy as np

_SRC = Path(__file__).resolve().parents[1] / "src"
if _SRC.is_dir() and str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from deltawkrel.certified_bounds import certified_robustness_interval  # noqa: E402
from deltawkrel.sdp import SWITCH_GENERALIZED_ROBUSTNESS_REFERENCE  # noqa: E402
from deltawkrel.switch_models import ideal_quantum_switch_process  # noqa: E402


def _json_safe(obj):
    import math

    if isinstance(obj, dict):
        return {k: _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return _json_safe(obj.tolist())
    if isinstance(obj, (np.floating, np.integer)):
        obj = float(obj)
    if isinstance(obj, float) and not math.isfinite(obj):
        return None
    return obj


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--outdir", default="results", help="Output directory.")
    parser.add_argument(
        "--solvers",
        default="SCS,CLARABEL,MOSEK",
        help="Comma-separated solver list for the interval table.",
    )
    parser.add_argument("--eps", type=float, default=1e-8, help="SCS tolerance.")
    parser.add_argument("--max-iters", type=int, default=100_000, help="SCS iteration limit.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    solvers = tuple(s.strip() for s in args.solvers.split(",") if s.strip())
    interval = certified_robustness_interval(
        ideal_quantum_switch_process(),
        solvers=solvers,
        eps=args.eps,
        max_iters=args.max_iters,
    )
    report = {
        "status": "certified_bounds",
        "title": "Certified interval for ideal-switch generalized robustness",
        "R_g_interval": {
            "lower": interval.R_g_lower,
            "upper": interval.R_g_upper,
            "width": interval.width,
            "published_reference": SWITCH_GENERALIZED_ROBUSTNESS_REFERENCE,
            "contains_published_reference": bool(
                interval.R_g_lower - 2e-4
                <= SWITCH_GENERALIZED_ROBUSTNESS_REFERENCE
                <= interval.R_g_upper + 2e-4
            ),
        },
        "solver_consensus_spread": interval.consensus_spread,
        "solver_table": [asdict(row) for row in interval.solver_table],
        "interpretation": (
            "This is a numerical certificate bracket, not only an 'optimal' flag: "
            "the dual value lower-bounds R_g, the primal value upper-bounds R_g, "
            "and the displayed width is the remaining primal-dual uncertainty."
        ),
    }
    out_path = outdir / "certified_bounds_report.json"
    out_path.write_text(
        json.dumps(_json_safe(report), indent=2, ensure_ascii=False, allow_nan=False),
        encoding="utf-8",
    )

    print(
        "R_g interval = "
        f"[{interval.R_g_lower:.12f}, {interval.R_g_upper:.12f}] "
        f"(width {interval.width:.2e})"
    )
    for row in interval.solver_table:
        print(
            f"  {row.solver:8s} available={row.available} status={row.status} "
            f"lower={row.R_g_lower:.12g} upper={row.R_g_upper:.12g} note={row.note}"
        )
    print(f"wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
