"""Shared test configuration for the ΔW/K_rel reproducibility package.

Forces a headless matplotlib backend so that plotting code paths can be
exercised in CI, and makes the `scripts/` modules importable by plain name
(pyproject already adds `scripts` to pythonpath; this file only pins the
backend before any test imports pyplot).
"""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg", force=True)
