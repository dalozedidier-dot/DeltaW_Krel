from __future__ import annotations

import json

import numpy as np
import pytest

import run_switch_dephasing_scan as scan
from deltawkrel.switch_models import (
    biased_coherent_switch_process,
    dephased_switch_process,
    ideal_quantum_switch_process,
    partially_dephased_switch_process,
    switch_branch_process,
    switch_white_noise_process,
    white_visibility_switch_process,
)


def test_partially_dephased_switch_interpolates_endpoints():
    W0 = partially_dephased_switch_process(0.0)
    W1 = partially_dephased_switch_process(1.0)
    assert np.allclose(W0, ideal_quantum_switch_process())
    assert np.allclose(W1, dephased_switch_process())
    assert np.isclose(np.trace(partially_dephased_switch_process(0.37)), 4.0)


def test_partially_dephased_switch_damps_only_coherence_block():
    lam = 0.25
    expected = (1.0 - lam) * ideal_quantum_switch_process() + lam * dephased_switch_process()
    assert np.allclose(partially_dephased_switch_process(lam), expected)
    with pytest.raises(ValueError):
        partially_dephased_switch_process(-0.1)
    with pytest.raises(ValueError):
        partially_dephased_switch_process(1.1)


def test_white_visibility_switch_interpolates_white_noise_and_ideal_switch():
    assert np.allclose(white_visibility_switch_process(0.0), switch_white_noise_process())
    assert np.allclose(white_visibility_switch_process(1.0), ideal_quantum_switch_process())
    assert np.isclose(np.trace(white_visibility_switch_process(0.42)), 4.0)
    with pytest.raises(ValueError):
        white_visibility_switch_process(-0.01)


def test_biased_coherent_switch_interpolates_fixed_order_endpoints():
    assert np.allclose(biased_coherent_switch_process(1.0), switch_branch_process("AB"))
    assert np.allclose(biased_coherent_switch_process(0.0), switch_branch_process("BA"))
    assert np.allclose(biased_coherent_switch_process(0.5), ideal_quantum_switch_process())
    assert np.isclose(np.trace(biased_coherent_switch_process(0.2)), 4.0)
    with pytest.raises(ValueError):
        biased_coherent_switch_process(1.01)


def test_parse_lambdas_validates_bounds():
    assert scan.parse_lambdas("0, 0.25,1") == [0.0, 0.25, 1.0]
    with pytest.raises(ValueError):
        scan.parse_lambdas("")
    with pytest.raises(ValueError):
        scan.parse_lambdas("0,1.5")


def test_summarize_curve_identifies_expected_shape():
    rows = [
        {"lambda_dephasing": 0.0, "generalized_robustness": 0.545351},
        {"lambda_dephasing": 0.5, "generalized_robustness": 0.167483},
        {"lambda_dephasing": 1.0, "generalized_robustness": 3.3e-10},
    ]
    summary = scan.summarize_curve(rows)
    assert summary["monotone_nonincreasing"] is True
    assert summary["positive_for_sampled_lambda_below_1"] is True
    assert summary["classical_endpoint_zero"] is True


def test_summarize_curve_detects_failed_shape_checks():
    rows = [
        {"lambda_dephasing": 0.0, "generalized_robustness": 0.1},
        {"lambda_dephasing": 0.5, "generalized_robustness": 0.2},
    ]
    summary = scan.summarize_curve(rows)
    assert summary["monotone_nonincreasing"] is False
    assert summary["classical_endpoint_zero"] is False


def test_json_safe_replaces_nested_nonfinite_values():
    safe = scan._json_safe({"a": (float("nan"), [float("inf"), 1.0])})
    assert safe == {"a": [None, [None, 1.0]]}


def test_write_outputs_strict_json_and_csv(tmp_path):
    rows = [
        {
            "lambda_dephasing": 0.0,
            "generalized_robustness": 0.545351,
            "status": "optimal",
            "solver": "SCS",
            "solver_version": "x",
            "cvxpy_version": "y",
            "num_iters": 10.0,
            "solve_time_s": 0.1,
            "equality_residual_fro": 1e-11,
            "subspace_residual_fro": 1e-11,
            "min_eig_W_AB": -1e-10,
            "min_eig_W_BA": -1e-10,
            "witness_value": 0.545351,
            "witness_certificate_gap": float("nan"),
        },
        {
            "lambda_dephasing": 1.0,
            "generalized_robustness": 0.0,
            "status": "optimal",
            "solver": "SCS",
            "solver_version": "x",
            "cvxpy_version": "y",
            "num_iters": 10.0,
            "solve_time_s": 0.1,
            "equality_residual_fro": 1e-11,
            "subspace_residual_fro": 1e-11,
            "min_eig_W_AB": -1e-10,
            "min_eig_W_BA": -1e-10,
            "witness_value": 0.0,
            "witness_certificate_gap": 1e-10,
        },
    ]
    json_path, csv_path = scan.write_outputs(rows, tmp_path)
    text = json_path.read_text(encoding="utf-8")
    assert "NaN" not in text
    report = json.loads(text)
    assert report["summary"]["classical_endpoint_zero"] is True
    assert csv_path.read_text(encoding="utf-8").startswith("lambda_dephasing,")


def test_dephasing_main_uses_cli_lambdas_and_writes_outputs(tmp_path, monkeypatch):
    def fake_scan_dephasing_curve(lambdas, solver="SCS", eps=1e-8, max_iters=200_000):
        return [
            {
                "lambda_dephasing": lam,
                "generalized_robustness": 1.0 - lam,
                "status": "optimal",
                "solver": solver,
                "solver_version": "x",
                "cvxpy_version": "y",
                "num_iters": 1.0,
                "solve_time_s": 0.1,
                "equality_residual_fro": 1e-11,
                "subspace_residual_fro": 1e-11,
                "min_eig_W_AB": -1e-10,
                "min_eig_W_BA": -1e-10,
                "witness_value": 1.0 - lam,
                "witness_certificate_gap": 1e-10,
            }
            for lam in lambdas
        ]

    monkeypatch.setattr(scan, "scan_dephasing_curve", fake_scan_dephasing_curve)
    code = scan.main(
        [
            "--lambdas",
            "0,1",
            "--outdir",
            str(tmp_path),
            "--solver",
            "FAKE",
        ]
    )
    assert code == 0
    report = json.loads((tmp_path / "switch_dephasing_scan.json").read_text(encoding="utf-8"))
    assert [row["lambda_dephasing"] for row in report["rows"]] == [0.0, 1.0]
