"""Public entry point for the full realistic tomography stress test.

This wrapper keeps the manuscript/site command focused on the experimental
meaning while reusing the tested implementation in ``full_tomography_simulation``.
"""
from __future__ import annotations

from full_tomography_simulation import main


if __name__ == "__main__":  # pragma: no cover - exercised through the target module
    raise SystemExit(main())
