#!/usr/bin/env python
r"""A4 - finite-count statistics of the single-direction estimator.

Extracts the fixed witness S* (and its admissible projection K_rel), then:

  1. confirms Var(lambda_hat) ~ 1/N by multinomial simulation (single-parameter
     scaling: one functional, not the D^2 process);
  2. maps N_required(lambda_target, visibility) to certify causal
     nonseparability at alpha = 0.01 with the single scalar;
  3. runs the false-positive test under calibrated-noise drift, showing K_rel
     stays at the nominal rate while the raw witness inflates.

Outputs under results/: finite_count_report.json, finite_count.png.
"""
from __future__ import annotations

import json
import os
import sys
import tempfile
from pathlib import Path

import numpy as np

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "mpl-fc"))
_SRC = Path(__file__).resolve().parents[1] / "src"
if _SRC.is_dir() and str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from deltawkrel.certified_witness import (  # noqa: E402
    admissible_direction, switch_generalized_robustness_witness)
from deltawkrel.finite_count import (  # noqa: E402
    copies_to_certify, false_positive_under_drift, lambda_estimator_scaling)
from deltawkrel.switch_models import (  # noqa: E402
    dephased_switch_process, ideal_quantum_switch_process,
    partially_dephased_switch_process, switch_white_noise_process)


def _json_safe(obj):
    import math
    if isinstance(obj, dict): return {k: _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)): return [_json_safe(v) for v in obj]
    if isinstance(obj, np.ndarray): return _json_safe(obj.tolist())
    if isinstance(obj, (np.floating, np.integer)): obj = float(obj)
    if isinstance(obj, float) and not math.isfinite(obj): return None
    return obj


def main() -> None:
    outdir = Path("results"); outdir.mkdir(exist_ok=True)
    W_switch = ideal_quantum_switch_process(); D = W_switch.shape[0]
    cert = switch_generalized_robustness_witness(W_switch, eps=1e-9)
    S_raw = cert.S

    # Signal axis (traceless): the falsification direction we test along.
    signal_axis = dephased_switch_process() - W_switch
    sig_hat = signal_axis / np.linalg.norm(signal_axis)

    # Calibrated-noise directions MUST be quasi-orthogonal to the signal
    # (config: quasi_orthogonality, calibration_only_pre_registered). Noise
    # *along* the signal is indistinguishable from signal for any witness; we
    # only claim immunity to the orthogonal calibrated components. We build the
    # worst-case confuser: the part of the raw witness NOT along the signal,
    # i.e. exactly the calibrated direction most likely to fool a naive witness.
    tr = lambda A, B: float(np.sum(A * B))
    N_conf = S_raw - tr(S_raw, sig_hat) * sig_hat
    N_conf = N_conf - (np.trace(N_conf) / D) * np.eye(D)     # remove normalization drift
    N_conf = N_conf / np.linalg.norm(N_conf)
    calibrated = [np.eye(D), N_conf]                         # both orthogonal to the signal
    adm = admissible_direction(S_raw, calibrated, signal_axis)
    K_rel = adm.K_rel
    # Express drift in signal-scale units so a drift of x is x times a full
    # falsification-scale perturbation (physically comparable magnitude).
    drift_dir_for_fp = N_conf * float(np.linalg.norm(signal_axis))

    # 1. 1/N scaling (optimal witness S*; single-functional estimator).
    scaling = lambda_estimator_scaling(S_raw, partially_dephased_switch_process,
                                       lambda_ref=0.1)
    ratio = scaling.var_lambda_empirical / scaling.var_lambda_analytic

    # 2. N_required(lambda_target, visibility).
    lam_targets = np.linspace(0.0, 0.65, 14)
    visibilities = [1.0, 0.9, 0.8, 0.7]
    surface = {}
    for v in visibilities:
        def fam_v(lam, v=v):
            return v * partially_dephased_switch_process(lam) + (1 - v) * switch_white_noise_process()
        surface[f"v={v:.1f}"] = [copies_to_certify(S_raw, fam_v, float(lt)) for lt in lam_targets]

    # 3. false positives under calibrated-noise drift (on a separable process).
    W_sep = dephased_switch_process()          # causally separable, witness < 0
    fp = false_positive_under_drift(K_rel, S_raw, W_sep, drift_dir_for_fp,
                                    drift_grid=(0.0, 0.2, 0.4, 0.6, 0.8, 1.0),
                                    N=5000, n_trials=800)

    report = {
        "title": "Finite-count single-direction estimator (shot-noise floor)",
        "process_free_parameters": int(D * D),
        "scalars_measured": 1,
        "krel_cos_angle_to_raw": adm.cos_angle_to_S,
        "one_over_N_scaling": {
            "N_grid": scaling.N_grid,
            "var_lambda_analytic": scaling.var_lambda_analytic,
            "var_lambda_empirical": scaling.var_lambda_empirical,
            "empirical_over_analytic_ratio": ratio,
            "scaling_confirmed": bool(np.all(np.abs(ratio - 1.0) < 0.25)),
            "slope_rho": scaling.slope_rho,
            "per_copy_variance_ref": scaling.per_copy_variance_ref,
        },
        "N_required_surface": {
            "lambda_targets": lam_targets,
            "visibilities": visibilities,
            "N_required": surface,
            "alpha": 0.01,
            "boundary": "witness value 0 (causally separable side)",
        },
        "false_positive_under_calibrated_drift": {
            "drift_grid": fp.drift_grid,
            "fp_rate_krel": fp.fp_rate_krel,
            "fp_rate_raw_witness": fp.fp_rate_raw,
            "krel_signal_shift": fp.krel_signal_shift,
            "raw_signal_shift": fp.raw_signal_shift,
            "interpretation": "K_rel is orthogonal to the drift axis: its signal does not move "
                              "and its FP rate stays near alpha; the raw witness drifts positive "
                              "and inflates FP.",
        },
        "idealisation_note": "Shot-noise floor of a direct witness measurement on the normalised "
                             "process; a lower bound on achievable variance. Calibrated tomography "
                             "(realistic_tomography_pipeline.py) can only increase N, never decrease it.",
    }
    (outdir / "finite_count_report.json").write_text(
        json.dumps(_json_safe(report), indent=2, ensure_ascii=False, allow_nan=False),
        encoding="utf-8")

    try:
        import matplotlib; matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        green, gold, red = "#2f6b4f", "#b8892b", "#8a2b2b"
        fig, ax = plt.subplots(1, 3, figsize=(13.5, 4.2))
        ax[0].loglog(scaling.N_grid, scaling.var_lambda_analytic, color=green, lw=2,
                     label=r"analytic $\propto 1/N$")
        ax[0].loglog(scaling.N_grid, scaling.var_lambda_empirical, "o", color=gold,
                     label="multinomial sim")
        ax[0].set_xlabel("copies $N$"); ax[0].set_ylabel(r"$\mathrm{Var}(\hat\lambda)$")
        ax[0].set_title("single-parameter scaling"); ax[0].grid(alpha=.25, which="both")
        ax[0].legend(fontsize=8)
        for v, row in surface.items():
            ax[1].semilogy(lam_targets, row, marker=".", label=v)
        ax[1].set_xlabel(r"$\lambda_{\rm target}$"); ax[1].set_ylabel("copies to certify")
        ax[1].set_title(r"$N_{\rm required}(\lambda,\ {\rm visibility})$")
        ax[1].grid(alpha=.25, which="both"); ax[1].legend(fontsize=8, title="visibility")
        ax[2].plot(fp.drift_grid, fp.fp_rate_krel, "o-", color=green, label=r"$K_{\rm rel}$ (admissible)")
        ax[2].plot(fp.drift_grid, fp.fp_rate_raw, "s--", color=red, label=r"$S^\star$ (raw)")
        ax[2].axhline(0.01, color="0.5", lw=.8, ls=":", label=r"$\alpha=0.01$")
        ax[2].set_xlabel("calibrated-noise drift"); ax[2].set_ylabel("false-positive rate")
        ax[2].set_title("relational immunity"); ax[2].grid(alpha=.25); ax[2].legend(fontsize=8)
        fig.suptitle("A4 - finite-count single-direction estimator", y=1.02, fontsize=12)
        fig.tight_layout(); fig.savefig(outdir / "finite_count.png", dpi=150, bbox_inches="tight")
        plt.close(fig)
    except Exception as exc:  # pragma: no cover
        print(f"(figure skipped: {exc})")

    print(f"1/N scaling confirmed: {report['one_over_N_scaling']['scaling_confirmed']} "
          f"(emp/analytic ratios {np.round(ratio,3)})")
    print(f"N to certify lambda=0 at v=1.0: {surface['v=1.0'][0]:.0f} copies "
          f"(vs {D*D} process params)")
    print(f"FP rate  K_rel: {np.round(fp.fp_rate_krel,3)}  | raw S*: {np.round(fp.fp_rate_raw,3)}")
    print("wrote results/finite_count_report.json, finite_count.png")


if __name__ == "__main__":
    main()
