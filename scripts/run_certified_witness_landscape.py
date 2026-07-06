#!/usr/bin/env python
r"""Run one fixed certified witness across three quantum-switch families.

The script extracts the dual-optimal witness ``S*`` once at the ideal switch,
then applies the same preregistered scalar to:

* control dephasing;
* white-noise visibility;
* coherent order bias.

It also verifies the lower-bound relation against the full SDP on a coarse grid.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from pathlib import Path

import numpy as np

os.environ.setdefault(
    "MPLCONFIGDIR",
    str(Path(tempfile.gettempdir()) / "deltawkrel_certified_witness_landscape_mpl"),
)

_SRC = Path(__file__).resolve().parents[1] / "src"
if _SRC.is_dir() and str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from deltawkrel.certified_witness import (  # noqa: E402
    certify_family,
    switch_generalized_robustness_witness,
)
from deltawkrel.sdp import (  # noqa: E402
    SWITCH_GENERALIZED_ROBUSTNESS_REFERENCE,
    solve_switch_generalized_robustness,
)
from deltawkrel.switch_models import (  # noqa: E402
    biased_coherent_switch_process,
    ideal_quantum_switch_process,
    partially_dephased_switch_process,
    white_visibility_switch_process,
)


FAMILIES = {
    "control_dephasing": {
        "fn": partially_dephased_switch_process,
        "reference": 0.0,
        "xlabel": r"control dephasing $\lambda$",
        "verify": [0.0, 0.2, 0.4, 0.6, 0.8, 1.0],
    },
    "white_visibility": {
        "fn": white_visibility_switch_process,
        "reference": 1.0,
        "xlabel": r"visibility $v$",
        "verify": [1.0, 0.8, 0.6, 0.4, 0.2, 0.0],
    },
    "order_bias": {
        "fn": biased_coherent_switch_process,
        "reference": 0.5,
        "xlabel": r"order bias $q$",
        "verify": [0.0, 0.2, 0.4, 0.5, 0.6, 0.8, 1.0],
    },
}


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
        "--dense-grid",
        type=int,
        default=41,
        help="Number of points for the cheap witness curves.",
    )
    parser.add_argument("--eps", type=float, default=1e-9, help="SCS tolerance for the reference SDP.")
    parser.add_argument(
        "--verify-eps",
        type=float,
        default=1e-7,
        help="SCS tolerance for the coarse lower-bound verification grid.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    W_switch = ideal_quantum_switch_process()
    cert = switch_generalized_robustness_witness(W_switch, eps=args.eps)
    benchmark_error = abs(cert.R_g - SWITCH_GENERALIZED_ROBUSTNESS_REFERENCE)
    print(
        f"reference witness S*: R_g={cert.R_g:.6f} "
        f"(err {benchmark_error:.1e}, tightness {cert.tightness_gap:.1e})"
    )

    dense = np.linspace(0.0, 1.0, int(args.dense_grid))
    report = {
        "status": "certified_witness_landscape",
        "title": "One preregistered witness certifies three switch families",
        "reference_witness": {
            "R_g": cert.R_g,
            "published_reference": SWITCH_GENERALIZED_ROBUSTNESS_REFERENCE,
            "benchmark_error": benchmark_error,
            "tightness_gap": cert.tightness_gap,
            "status": cert.status,
        },
        "families": {},
    }
    csv_rows = ["family,param,witness_S,R_g_sdp"]
    fig_data = {}

    for name, spec in FAMILIES.items():
        fam = spec["fn"]
        fc = certify_family(cert.S, fam, dense, spec["reference"], cert.R_g)

        verify = []
        for x in spec["verify"]:
            process = fam(float(x))
            rg = solve_switch_generalized_robustness(process, eps=args.verify_eps).objective_value
            wv = float(cert.value(process))
            verify.append(
                {
                    "param": float(x),
                    "witness_S": wv,
                    "R_g_sdp": float(rg),
                    "gap": float(rg - wv),
                }
            )
            csv_rows.append(f"{name},{x:.4f},{wv:.8f},{float(rg):.8f}")
        lower_bound_holds = bool(min(v["gap"] for v in verify) >= -1e-6)

        report["families"][name] = {
            "is_affine": fc.is_affine,
            "affinity_residual": fc.affinity_residual,
            "reference_param": fc.reference_param,
            "reference_value": fc.reference_value,
            "tight_at_reference": bool(abs(fc.reference_value - cert.R_g) < 1e-6),
            "slope": fc.slope,
            "certified_crossings": fc.certified_crossings,
            "certified_nonseparable_region": list(fc.certified_region),
            "single_scalar_injective": fc.single_scalar_injective,
            "lower_bound_holds_on_grid": lower_bound_holds,
            "note": fc.note,
            "verification_grid": verify,
        }
        fig_data[name] = {
            "x": fc.params,
            "y": fc.witness_values,
            "vx": [v["param"] for v in verify],
            "vy": [v["R_g_sdp"] for v in verify],
            "region": fc.certified_region,
            "xlabel": spec["xlabel"],
            "affine": fc.is_affine,
        }
        tag = "affine, one-sided" if fc.is_affine else "nonlinear, two-sided"
        region = tuple(round(r, 3) for r in fc.certified_region)
        print(
            f"  {name:18s} [{tag}] affinity_res={fc.affinity_residual:.1e} "
            f"region={region} lower_bound={lower_bound_holds}"
        )

    (outdir / "certified_witness_landscape.json").write_text(
        json.dumps(_json_safe(report), indent=2, ensure_ascii=False, allow_nan=False),
        encoding="utf-8",
    )
    (outdir / "certified_witness_landscape.csv").write_text(
        "\n".join(csv_rows) + "\n",
        encoding="utf-8",
    )

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.3))
        green, gold, red = "#2f6b4f", "#b8892b", "#8a2b2b"
        for ax, (name, d) in zip(axes, fig_data.items()):
            ax.plot(
                d["x"],
                d["y"],
                color=green,
                lw=2.1,
                label=r"$\mathrm{Tr}(S^\star W)$ certified bound",
            )
            ax.scatter(d["vx"], d["vy"], color=gold, s=34, zorder=5, label=r"$R_g$ full SDP")
            ax.axhline(0.0, color="0.5", lw=0.8)
            lo, hi = d["region"]
            ax.axvspan(lo, hi, color=green, alpha=0.08)
            ax.axvline(lo, color=red, ls="--", lw=1.1)
            ax.axvline(hi, color=red, ls="--", lw=1.1)
            kind = "affine / one-sided" if d["affine"] else "nonlinear / two-sided"
            ax.set_title(f"{name}\n({kind})", fontsize=10)
            ax.set_xlabel(d["xlabel"])
            ax.grid(alpha=0.22)
        axes[0].set_ylabel("robustness / witness value")
        axes[0].legend(fontsize=8, framealpha=0.95, loc="upper right")
        fig.suptitle("One preregistered witness certifies three switch families", fontsize=12, y=1.02)
        fig.tight_layout()
        fig.savefig(outdir / "certified_witness_landscape.png", dpi=150, bbox_inches="tight")
        plt.close(fig)
    except Exception as exc:  # pragma: no cover
        print(f"(figure skipped: {exc})")

    print("wrote certified_witness_landscape.json, .csv, .png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
