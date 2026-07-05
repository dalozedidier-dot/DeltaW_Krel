#!/usr/bin/env python
"""Freeze the preregistration config into a deterministic lock file."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config/config_preregistration.json")
    parser.add_argument("--out", default="results/preregistration_lock.json")
    args = parser.parse_args()

    config_path = Path(args.config)
    out_path = Path(args.out)
    config = json.loads(config_path.read_text(encoding="utf-8"))

    lock = {
        "status": "locked",
        "config_path": config_path.as_posix(),
        "config_sha256": sha256_file(config_path),
        "scenario": config.get("scenario"),
        "trace_convention": config.get("trace_convention"),
        "thresholds": config.get("thresholds", {}),
        "solver": config.get("solver", {}),
        "witness": config.get("witness", {}),
        "confirmatory_note": (
            "A confirmatory analysis should verify this hash before using the "
            "thresholds or witness-selection rules."
        ),
    }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps(lock, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"wrote {out_path}")
    print(f"config_sha256={lock['config_sha256']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
