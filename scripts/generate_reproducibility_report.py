#!/usr/bin/env python
"""Generate a compact reproducibility report from repository artifacts."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def load_json(path: Path) -> dict | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def status_line(label: str, path: Path, legacy_paths: tuple[Path, ...] = ()) -> str:
    """Return a stable artifact-status line.

    The canonical location is reported first.  Legacy locations are only used as
    a transition aid so older CI bundles remain readable without hiding that the
    repository standard has moved to ``results/``.
    """
    if path.exists():
        status = "present"
    elif any(legacy.exists() for legacy in legacy_paths):
        status = "legacy-present"
    else:
        status = "missing"
    line = f"- {label}: {status} (`{path.as_posix()}`)"
    if status == "legacy-present":
        found = [legacy.as_posix() for legacy in legacy_paths if legacy.exists()]
        line += f"; legacy artifact found at `{found[0]}`"
    return line


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="results/reproducibility_report.md")
    args = parser.parse_args()

    out_path = Path(args.out)
    sdp = load_json(Path("results/sdp_validation_report.json"))
    lock = load_json(Path("results/preregistration_lock.json"))
    manifest = load_json(Path("MANIFEST.sha256.json")) or {}

    lines: list[str] = [
        "# DeltaW/K_rel reproducibility report",
        "",
        "This report is generated from repository artifacts. It distinguishes",
        "implemented evidence from explicit submission blockers.",
        "",
        "## Artifact status",
        "",
        status_line("Preregistration config", Path("config/config_preregistration.json")),
        status_line("Preregistration lock", Path("results/preregistration_lock.json")),
        status_line("SDP validation report", Path("results/sdp_validation_report.json")),
        status_line(
            "Monte Carlo smoke output",
            Path("results/mc_smoke/monte_carlo_power_results.json"),
            legacy_paths=(Path("monte_carlo_outputs_control/monte_carlo_power_results.json"),),
        ),
        status_line(
            "Micro-tomography smoke output",
            Path("results/micro_smoke/micro_tomography_power.csv"),
            legacy_paths=(Path("outputs/micro_tomography_power.csv"),),
        ),
        status_line(
            "Realistic tomography bridge output",
            Path("results/realistic_tomography_smoke/realistic_tomography_report.json"),
        ),
        status_line(
            "Full tomography stress output",
            Path("results/full_tomography_smoke/full_tomography_report.json"),
        ),
        status_line(
            "Cao 2023 public-count verification",
            Path("results/external/cao2023_sdi/cao2023_sdi_report.json"),
        ),
        status_line("Claim/evidence matrix", Path("docs/CLAIM_EVIDENCE_MATRIX.md")),
        "",
        "## Preregistration lock",
        "",
    ]

    if lock is None:
        lines.append("- status: missing")
    else:
        lines.extend(
            [
                f"- status: {lock.get('status')}",
                f"- scenario: {lock.get('scenario')}",
                f"- trace convention: {lock.get('trace_convention')}",
                f"- config sha256: `{lock.get('config_sha256')}`",
            ]
        )

    lines.extend(["", "## SDP validation", ""])
    if sdp is None:
        lines.append("- status: missing")
        benchmark_passed = False
    else:
        lines.append(f"- status: {sdp.get('status')}")
        blocker = sdp.get("ideal_quantum_switch_benchmark", {})
        benchmark_passed = bool(blocker.get("benchmark_passed", False))
        lines.append(
            "- ideal quantum switch implemented: "
            f"{bool(blocker.get('implemented', False))}"
        )
        lines.append(
            "- ideal quantum switch submission blocker: "
            f"{bool(blocker.get('submission_blocker', True))}"
        )
        if "computed_generalized_robustness" in blocker:
            lines.append(
                "- switch generalized robustness: "
                f"{blocker.get('computed_generalized_robustness')} "
                f"(reference {blocker.get('reference_generalized_robustness')}, "
                f"passed={benchmark_passed})"
            )
        targets = sdp.get("targets", {})
        for name, diag in targets.items():
            lines.append(
                f"- {name}: {diag.get('status')} "
                f"(objective={diag.get('objective_value')}, solver={diag.get('solver')})"
            )

    lines.extend(
        [
            "",
            "## Manifest",
            "",
            f"- manifest entries: {len(manifest)}",
            "- validation command: `python scripts/validate_manifest.py`",
            "",
            "## Submission status",
            "",
            "The repository supports toy/geometric simulations (methodological",
            "control bench, NOT experimental validation), the micro-tomography",
            "proof of concept, a finite-count realistic tomography simulation",
            "bridge, a full simulated tomography stress test, process-matrix",
            "projector tests, K_CS SDP validation, and one public semi-DI",
            "external count-file verification. The external public-data pilot is",
            "not yet a DeltaW/K_rel process-matrix reanalysis.",
            "The ideal quantum switch is implemented in the",
            "Araújo et al. (NJP 17, 102001 (2015)) convention; its generalized",
            "robustness benchmark (reference 0.5454) is "
            + ("REPRODUCED" if benchmark_passed else "NOT YET REPRODUCED — submission blocker")
            + ". See docs/CONVENTIONS.md for the official convention and the",
            "equation-to-code audit table.",
            "",
        ]
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
