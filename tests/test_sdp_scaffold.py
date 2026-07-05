import pytest

cvxpy = pytest.importorskip("cvxpy")

from deltawkrel.projectors import ProcessDims, white_noise_process
from deltawkrel.switch_models import (
    fixed_order_A_before_B_process,
    fixed_order_B_before_A_process,
    causally_separable_mixture,
)
from deltawkrel.sdp import solve_cs_robustness


def test_white_noise_has_zero_cs_robustness():
    dims = ProcessDims(2, 2, 2, 2)
    W = white_noise_process(dims)
    diag = solve_cs_robustness(W, dims=dims, solver="CLARABEL")
    assert diag.status in {"optimal", "optimal_inaccurate"}
    assert diag.objective_value < 1e-5
    assert diag.omega_white == pytest.approx(0.0, abs=1e-8)


def test_fixed_order_processes_have_zero_cs_robustness():
    dims = ProcessDims(2, 2, 2, 2)
    for W in (
        fixed_order_A_before_B_process(dims),
        fixed_order_B_before_A_process(dims),
        causally_separable_mixture(dims, q=0.37),
    ):
        diag = solve_cs_robustness(W, dims=dims, solver="CLARABEL")
        assert diag.status in {"optimal", "optimal_inaccurate"}
        assert diag.objective_value < 1e-5
