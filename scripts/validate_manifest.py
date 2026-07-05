#!/usr/bin/env python
"""Validate repository files against MANIFEST.sha256.json."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk.replace(b"\r\n", b"\n"))
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest",
        default="MANIFEST.sha256.json",
        help="Path to the SHA-256 manifest, relative to the repository root.",
    )
    parser.add_argument(
        "--root",
        default=".",
        help="Repository root used to resolve manifest paths.",
    )
    args = parser.parse_args()

    root = Path(args.root).resolve()
    manifest_path = (root / args.manifest).resolve()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    missing: list[str] = []
    mismatched: list[tuple[str, str, str]] = []

    for relative_path, expected_hash in sorted(manifest.items()):
        path = root / relative_path
        if not path.exists():
            missing.append(relative_path)
            continue
        actual_hash = sha256_file(path)
        if actual_hash != expected_hash:
            mismatched.append((relative_path, expected_hash, actual_hash))

    if missing or mismatched:
        if missing:
            print("Missing files:")
            for relative_path in missing:
                print(f"  - {relative_path}")
        if mismatched:
            print("Hash mismatches:")
            for relative_path, expected_hash, actual_hash in mismatched:
                print(f"  - {relative_path}")
                print(f"    expected: {expected_hash}")
                print(f"    actual:   {actual_hash}")
        return 1

    print(f"Validated {len(manifest)} files against {manifest_path.name}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
