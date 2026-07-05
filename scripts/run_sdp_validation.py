#!/usr/bin/env python
"""Run the K_CS SDP validation suite and export full diagnostics.

Two levels are validated and exported to results/sdp_validation_report.json:

1. Bipartite K_CS wiring (no future space): white-noise, fixed-order and
   causally separable mixture targets must all have zero robustness.
2. Ideal quantum switch benchmark (with global future F = Ft ⊗ Fc, convention
   of Araújo et al., NJP 17, 102001 (2015)): the generalized robustness must
   reproduce the published value ≈ 0.5454, and the control-dephased switch
   must be causally separable (robustness 0).

Every SDP solve exports extended diagnostics: solver + versions, iterations,
solve time, equality/subspace residuals, minimal eigenvalues of the cone
variables, and the dual causal-witness certificate value.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

# Allow running from a clean clone without `pip install -e .`.
_SRC = Path(__file__).resolve().parents[1] / "src"
if _SRC.is_dir() and str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from deltawkrel.projectors import ProcessDims, assert_trace_convention, white_noise_process
from deltawkrel.switch_models import (
    fixed_order_A_before_B_process,
    fixed_order_B_before_A_process,
    causally_separable_mixture,
    dephased_switch_process,
    ideal_quantum_switch_process,
)
from deltawkrel.sdp import (
    SWITCH_BENCHMARK_TOLERANCE,
    SWITCH_GENERALIZED_ROBUSTNESS_REFERENCE,
    solve_cs_robustness,
    solve_switch_generalized_robustness,
    solve_switch_random_robustness,
)


def _json_safe(obj):
    """Recursively replace non-finite floats by None for strict JSON export."""
    import math

    if isinstance(obj, dict):
        return {k: _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(v) for v in obj]
    if isinstance(obj, float) and not math.isfinite(obj):
        return None
    return obj


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
    report: dict = {
        "status": "sdp_validation_with_switch_benchmark",
        "dims": {"AI": dims.AI, "AO": dims.AO, "BI": dims.BI, "BO": dims.BO, "d_O": dims.d_O},
        "targets": {},
    }
    for name, W in targets.items():
        assert_trace_convention(W, dims)
        diag = solve_cs_robustness(W, dims=dims, solver="CLARABEL")
        report["targets"][name] = diag.to_dict()

    # --- Ideal quantum switch benchmark (global future space, D=64).
    W_switch = ideal_quantum_switch_process()
    gen = solve_switch_generalized_robustness(W_switch)
    rand = solve_switch_random_robustness(W_switch)
    dephased = solve_switch_generalized_robustness(dephased_switch_process())

    benchmark_error = abs(gen.objective_value - SWITCH_GENERALIZED_ROBUSTNESS_REFERENCE)
    benchmark_passed = (
        gen.status in {"optimal", "optimal_inaccurate"}
        and benchmark_error <= SWITCH_BENCHMARK_TOLERANCE
        and dephased.status in {"optimal", "optimal_inaccurate"}
        and abs(dephased.objective_value) <= 1e-6
    )

    report["ideal_quantum_switch_benchmark"] = {
        "implemented": True,
        "convention": (
            "Araújo, Branciard, Costa, Feix, Giarmatzi, Brukner, NJP 17, 102001 (2015); "
            "systems [AI, AO, BI, BO, F], F = F_target ⊗ F_control, control |+>, target |0>."
        ),
        "reference_generalized_robustness": SWITCH_GENERALIZED_ROBUSTNESS_REFERENCE,
        "tolerance": SWITCH_BENCHMARK_TOLERANCE,
        "computed_generalized_robustness": gen.objective_value,
        "benchmark_error": benchmark_error,
        "benchmark_passed": benchmark_passed,
        "submission_blocker": not benchmark_passed,
        "generalized_robustness_diagnostics": gen.to_dict(),
        "random_robustness_diagnostics": rand.to_dict(),
        "dephased_switch_diagnostics": dephased.to_dict(),
        "note": (
            "The control-dephased switch must be causally separable (robustness 0); "
            "the ideal switch must reproduce the published generalized robustness."
        ),
    }

    path = outdir / "sdp_validation_report.json"
    path.write_text(
        json.dumps(_json_safe(report), indent=2, ensure_ascii=False, allow_nan=False),
        encoding="utf-8",
    )
    print(f"wrote {path}")
    print(
        f"switch benchmark: R_g={gen.objective_value:.6f} "
        f"(reference {SWITCH_GENERALIZED_ROBUSTNESS_REFERENCE}, "
        f"error {benchmark_error:.2e}, passed={benchmark_passed})"
    )


if __name__ == "__main__":
    main()
