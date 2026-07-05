#!/usr/bin/env python
"""Regenerate MANIFEST.sha256.json from git-tracked files.

The manifest pins the reproducible sources of the repository. Volatile,
regenerated artifacts (results/, outputs/, the Monte Carlo output folder) and
the manifest itself are excluded so that `python scripts/validate_manifest.py`
stays green on a clean checkout and in CI.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
from pathlib import Path

EXCLUDED_PREFIXES = (
    "results/",
    "outputs/",
    "monte_carlo_outputs_control/",
)
EXCLUDED_FILES = {"MANIFEST.sha256.json"}


def git_executable() -> str:
    """Return a usable Git executable across local Codex and standard CI."""
    env_git = os.environ.get("GIT_EXE")
    if env_git and Path(env_git).exists():
        return env_git

    path_git = shutil.which("git")
    if path_git:
        return path_git

    bundled = (
        Path.home()
        / ".cache"
        / "codex-runtimes"
        / "codex-primary-runtime"
        / "dependencies"
        / "native"
        / "git"
        / "cmd"
        / "git.exe"
    )
    if bundled.exists():
        return str(bundled)

    raise RuntimeError(
        "Git executable not found. Install Git, add it to PATH, or set GIT_EXE."
    )


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk.replace(b"\r\n", b"\n"))
    return digest.hexdigest()


def list_tracked_files(root: Path) -> list[str]:
    output = subprocess.run(
        [git_executable(), "ls-files", "-z"],
        cwd=root,
        check=True,
        capture_output=True,
    ).stdout
    return [name.decode("utf-8") for name in output.split(b"\0") if name]


def build_manifest(root: Path) -> dict[str, str]:
    manifest: dict[str, str] = {}
    for relative_path in sorted(list_tracked_files(root)):
        if relative_path in EXCLUDED_FILES:
            continue
        if relative_path.startswith(EXCLUDED_PREFIXES):
            continue
        path = root / relative_path
        if not path.is_file():
            continue
        manifest[relative_path] = sha256_file(path)
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=".", help="Repository root.")
    parser.add_argument("--out", default="MANIFEST.sha256.json", help="Manifest path, relative to root.")
    args = parser.parse_args()

    root = Path(args.root).resolve()
    manifest = build_manifest(root)
    out_path = root / args.out
    out_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"wrote {out_path} ({len(manifest)} entries)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
