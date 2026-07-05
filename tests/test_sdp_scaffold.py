import numpy as np
import pytest

cvxpy = pytest.importorskip("cvxpy")

from deltawkrel.projectors import ProcessDims, white_noise_process
from deltawkrel.sdp import solve_cs_robustness_scaffold


def test_white_noise_has_zero_scaffold_robustness():
    dims = ProcessDims(2, 2, 2, 2)
    W = white_noise_process(dims)
    diag = solve_cs_robustness_scaffold(W, dims=dims, solver="CLARABEL")
    assert diag.status in {"optimal", "optimal_inaccurate"}
    assert diag.objective_value < 1e-5
    assert diag.omega_white == pytest.approx(0.0, abs=1e-8)
