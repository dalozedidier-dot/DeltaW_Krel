#!/usr/bin/env python
r"""Certified single-direction causal-witness analysis on the ideal switch.

This is the strongest single-author, no-lab result the DeltaW/K_rel program can
demonstrate: it fuses the SDP benchmark and the preregistered decision rule on
the *physical* switch process and certifies, from conic duality, that ONE fixed
scalar functional falsifies causal nonseparability along the whole dephasing
family.

Outputs (under results/):
  * certified_witness_report.json   full diagnostics + the certified theorem checks
  * certified_witness_curve.csv     lambda, witness signal, SDP robustness, gap
  * certified_witness_curve.png     the certified affine bound vs the SDP curve

What is proved / measured here:
  1. R_g(W_switch) reproduces the published 0.5454 benchmark (SDP layer).
  2. lambda -> Tr(S* W(lambda)) is affine to machine precision.
  3. Tr(S* W(lambda)) is a certified LOWER BOUND on the SDP R_g(lambda),
     tight at lambda=0 -> certified nonseparability threshold lambda*.
  4. A single expectation <S*> inverts lambda exactly (tomographic collapse:
     one scalar instead of D^2 = 4096 process parameters).
  5. K_rel = P_A(S*) is first-order immune to calibrated noise; the angle to S*
     quantifies the price of that relational robustness.
"""
from __future__ import annotations

import json
import os
import sys
import tempfile
from pathlib import Path

import numpy as np

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "mpl-cw"))

_SRC = Path(__file__).resolve().parents[1] / "src"
if _SRC.is_dir() and str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from deltawkrel.certified_witness import (  # noqa: E402
    admissible_direction,
    affine_witness_curve,
    switch_generalized_robustness_witness,
)
from deltawkrel.sdp import (  # noqa: E402
    SWITCH_GENERALIZED_ROBUSTNESS_REFERENCE,
    solve_switch_generalized_robustness,
)
from deltawkrel.switch_models import (  # noqa: E402
    dephased_switch_process,
    ideal_quantum_switch_process,
    partially_dephased_switch_process,
    switch_white_noise_process,
)


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


def target_depolarization_axis() -> np.ndarray:
    """A physical calibrated-noise axis distinct from the control-dephasing signal.

    Depolarizing pull toward the normalized white-noise process; a genuine
    nuisance direction the witness must be blind to.
    """
    return switch_white_noise_process() - ideal_quantum_switch_process()


def main() -> None:
    outdir = Path("results")
    outdir.mkdir(exist_ok=True)

    W_switch = ideal_quantum_switch_process()
    D = W_switch.shape[0]

    # 1. Dual-optimal causal witness at the reference (one high-accuracy SDP).
    cert = switch_generalized_robustness_witness(W_switch, eps=1e-9)
    benchmark_error = abs(cert.R_g - SWITCH_GENERALIZED_ROBUSTNESS_REFERENCE)

    # 2-3. Affine witness signal along the physical dephasing family (no SDP).
    dense = np.linspace(0.0, 1.0, 21)
    curve = affine_witness_curve(cert.S, partially_dephased_switch_process, dense)

    # Certified lower bound: solve the full SDP at a coarse verification grid.
    verify_lams = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
    sdp_vals, wit_vals, gaps = [], [], []
    for lam in verify_lams:
        W = partially_dephased_switch_process(lam)
        rg = solve_switch_generalized_robustness(W, eps=1e-7).objective_value
        wv = cert.value(W)
        sdp_vals.append(float(rg))
        wit_vals.append(float(wv))
        gaps.append(float(rg - wv))
    lower_bound_holds = bool(min(gaps) >= -1e-6)
    second_diff = np.diff(np.array(sdp_vals), 2)
    convex = bool(np.all(second_diff >= -1e-4))

    # 4. Single-scalar invertibility (tomographic collapse).
    inversion = []
    for lam in [0.1, 0.37, 0.6, 0.9]:
        y = cert.value(partially_dephased_switch_process(lam))
        lam_hat = curve.invert(y)
        inversion.append({"lambda": lam, "expectation": float(y),
                          "lambda_hat": float(lam_hat), "abs_error": float(abs(lam_hat - lam))})
    max_inv_err = max(r["abs_error"] for r in inversion)

    # 5. Relational direction K_rel = P_A(S*): immune to calibrated noise.
    I = np.eye(D)
    calibrated = [I, target_depolarization_axis()]
    signal_axis = dephased_switch_process() - W_switch
    adm = admissible_direction(cert.S, calibrated, signal_axis)
    K_curve = affine_witness_curve(adm.K_rel, partially_dephased_switch_process, dense)

    report = {
        "title": "Certified single-direction causal witness of the ideal quantum switch",
        "dimension": int(D),
        "process_free_parameters": int(D * D),
        "sdp_benchmark": {
            "R_g": cert.R_g,
            "published_reference": SWITCH_GENERALIZED_ROBUSTNESS_REFERENCE,
            "benchmark_error": benchmark_error,
            "witness_tightness_gap": cert.tightness_gap,
            "equality_residual": cert.equality_residual,
            "status": cert.status,
        },
        "affine_theorem": {
            "slope": curve.slope,
            "intercept": curve.intercept,
            "affinity_residual_max": curve.affinity_residual,
            "affinity_holds": bool(curve.affinity_residual < 1e-9),
            "tight_at_reference": bool(abs(curve.values[0] - cert.R_g) < 1e-6),
        },
        "certified_threshold": {
            "lambda_star": curve.zero_crossing,
            "meaning": "processes with lambda < lambda_star are certified causally nonseparable "
                       "by the single fixed witness S*.",
            "true_lambda_c_upper_bound": 1.0,
            "lower_bound_holds_on_grid": lower_bound_holds,
            "sdp_robustness_convex_on_grid": convex,
        },
        "tomographic_collapse": {
            "process_free_parameters": int(D * D),
            "scalars_measured_by_protocol": 1,
            "noiseless_lambda_recovery_max_error": max_inv_err,
            "inversion_examples": inversion,
        },
        "relational_direction_K_rel": {
            "calibrated_noise_directions": ["identity/normalization", "target depolarization"],
            "cos_angle_to_raw_witness": adm.cos_angle_to_S,
            "retained_norm_fraction": adm.retained_fraction,
            "noise_leakage_after_projection": adm.noise_leakage,
            "signal_response": adm.signal_response,
            "K_rel_affine_slope": K_curve.slope,
            "K_rel_zero_crossing": K_curve.zero_crossing,
            "interpretation": "K_rel keeps the signal component of the certificate while being "
                              "first-order blind to calibrated noise; the angle to S* is the price "
                              "of that robustness.",
        },
        "verification_grid": [
            {"lambda": l, "witness_S": w, "R_g_sdp": r, "gap_Rg_minus_witness": g}
            for l, w, r, g in zip(verify_lams, wit_vals, sdp_vals, gaps)
        ],
    }

    (outdir / "certified_witness_report.json").write_text(
        json.dumps(_json_safe(report), indent=2, ensure_ascii=False, allow_nan=False),
        encoding="utf-8",
    )

    # CSV of the dense witness curve.
    csv_lines = ["lambda,witness_S,K_rel_signal"]
    for lam, wS, wK in zip(curve.lambdas, curve.values, K_curve.values):
        csv_lines.append(f"{lam:.4f},{wS:.8f},{wK:.8f}")
    (outdir / "certified_witness_curve.csv").write_text("\n".join(csv_lines) + "\n", encoding="utf-8")

    # Figure.
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(7.2, 4.6))
        ax.plot(curve.lambdas, curve.values, color="#2f6b4f", lw=2.2,
                label=r"$\mathrm{Tr}(S^\star W(\lambda))$ (certified affine bound)")
        ax.scatter(verify_lams, sdp_vals, color="#b8892b", zorder=5, s=42,
                   label=r"$R_g(\lambda)$ full SDP (verification)")
        ax.axhline(0.0, color="0.5", lw=0.8)
        if np.isfinite(curve.zero_crossing):
            ax.axvline(curve.zero_crossing, color="#8a2b2b", ls="--", lw=1.4,
                       label=rf"certified threshold $\lambda^\star={curve.zero_crossing:.3f}$")
        ax.set_xlabel(r"control dephasing $\lambda$")
        ax.set_ylabel("causal robustness / witness value")
        ax.set_title("One preregistered scalar certifies the falsification curve")
        ax.legend(fontsize=8.5, framealpha=0.95)
        ax.grid(alpha=0.25)
        fig.tight_layout()
        fig.savefig(outdir / "certified_witness_curve.png", dpi=150)
        plt.close(fig)
    except Exception as exc:  # pragma: no cover
        print(f"(figure skipped: {exc})")

    print(f"R_g = {cert.R_g:.6f}  (ref {SWITCH_GENERALIZED_ROBUSTNESS_REFERENCE}, "
          f"err {benchmark_error:.1e}, tightness {cert.tightness_gap:.1e})")
    print(f"affinity residual = {curve.affinity_residual:.1e}  (affine iff ~0)")
    print(f"certified threshold lambda* = {curve.zero_crossing:.4f}  "
          f"| lower bound holds on grid = {lower_bound_holds} | R_g convex = {convex}")
    print(f"single-scalar lambda recovery max error = {max_inv_err:.1e}  "
          f"(vs {D*D} process parameters)")
    print(f"K_rel: cos(angle to S*) = {adm.cos_angle_to_S:.4f}, "
          f"noise leakage = {adm.noise_leakage}, retained = {adm.retained_fraction:.3f}")
    print("wrote results/certified_witness_report.json, .csv, .png")


if __name__ == "__main__":
    main()
