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


def status_line(label: str, path: Path) -> str:
    return f"- {label}: {'present' if path.exists() else 'missing'} (`{path.as_posix()}`)"


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
        status_line("Monte Carlo smoke output", Path("results/mc_smoke/monte_carlo_power_results.json")),
        status_line("Micro-tomography smoke output", Path("results/micro_smoke/micro_tomography_power.csv")),
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
    else:
        lines.append(f"- status: {sdp.get('status')}")
        blocker = sdp.get("ideal_quantum_switch_benchmark", {})
        lines.append(
            "- ideal quantum switch implemented: "
            f"{bool(blocker.get('implemented', False))}"
        )
        lines.append(
            "- ideal quantum switch submission blocker: "
            f"{bool(blocker.get('submission_blocker', True))}"
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
            "The repository currently supports toy/geometric simulations,",
            "micro-tomography proof of concept, process-matrix projector tests,",
            "and K_CS infrastructure validation. The ideal quantum-switch",
            "benchmark remains the decisive submission blocker until implemented",
            "and compared with a published reference value.",
            "",
        ]
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
