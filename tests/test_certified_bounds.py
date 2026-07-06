"""Tests for certified primal/dual robustness intervals."""
from __future__ import annotations

import pytest

cp = pytest.importorskip("cvxpy")

from deltawkrel.certified_bounds import certified_robustness_interval
from deltawkrel.sdp import SWITCH_GENERALIZED_ROBUSTNESS_REFERENCE
from deltawkrel.switch_models import ideal_quantum_switch_process


def test_certified_interval_brackets_switch_robustness():
    interval = certified_robustness_interval(
        ideal_quantum_switch_process(),
        solvers=("SCS",),
        eps=1e-8,
        max_iters=100_000,
    )
    assert interval.R_g_lower <= interval.R_g_upper
    assert interval.width < 1e-6
    assert interval.R_g_lower - 2e-4 <= SWITCH_GENERALIZED_ROBUSTNESS_REFERENCE
    assert SWITCH_GENERALIZED_ROBUSTNESS_REFERENCE <= interval.R_g_upper + 2e-4
    assert interval.solver_table[0].status in {"optimal", "optimal_inaccurate"}
