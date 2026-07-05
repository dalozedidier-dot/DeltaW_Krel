#!/usr/bin/env python
"""Run the minimal K_CS SDP validation suite and export diagnostics.

This script validates the projectors and the causally separable SDP wiring on
white-noise and fixed-order targets. It deliberately does not claim the ideal
quantum-switch benchmark; that benchmark remains a submission blocker until the
explicit switch process is implemented.
"""
from __future__ import annotations

import json
from pathlib import Path

from deltawkrel.projectors import ProcessDims, assert_trace_convention, white_noise_process
from deltawkrel.switch_models import (
    fixed_order_A_before_B_process,
    fixed_order_B_before_A_process,
    causally_separable_mixture,
    ideal_quantum_switch_process,
)
from deltawkrel.sdp import solve_cs_robustness


def main() -> None:
    outdir = Path("results")
    outdir.mkdir(exist_ok=True)
    dims = ProcessDims(2, 2, 2, 2)
    targets = {
        "white_noise": white_noise_process(dims),
        "fixed_order_A_before_B": fixed_order_A_before_B_process(dims),
        "fixed_order_B_before_A": fixed_order_B_before_A_process(dims),
        "causally_separable_mixture_q037": causally_separable_mixture(dims, q=0.37),
    }
    report = {
        "status": "infrastructure_validation_only",
        "dims": {"AI": dims.AI, "AO": dims.AO, "BI": dims.BI, "BO": dims.BO, "d_O": dims.d_O},
        "targets": {},
        "ideal_quantum_switch_benchmark": {
            "implemented": False,
            "submission_blocker": True,
            "note": "ideal_quantum_switch_process intentionally raises NotImplementedError until a published convention is implemented and benchmarked.",
        },
    }
    for name, W in targets.items():
        assert_trace_convention(W, dims)
        diag = solve_cs_robustness(W, dims=dims, solver="CLARABEL")
        report["targets"][name] = diag.to_dict()
    try:
        ideal_quantum_switch_process(dims)
    except NotImplementedError as exc:
        report["ideal_quantum_switch_benchmark"]["message"] = str(exc)
    path = outdir / "sdp_validation_report.json"
    path.write_text(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
