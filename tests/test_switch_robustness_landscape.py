from __future__ import annotations

import json

import pytest

import run_switch_robustness_landscape as landscape


def test_process_for_family_rejects_unknown_family():
    with pytest.raises(ValueError):
        landscape.process_for_family("unknown", 0.5)


def test_parse_grid_validates_unit_interval():
    assert landscape.parse_grid("0,0.5,1") == [0.0, 0.5, 1.0]
    with pytest.raises(ValueError):
        landscape.parse_grid("")
    with pytest.raises(ValueError):
        landscape.parse_grid("-0.1")


def test_summarize_rows_reports_family_shape():
    rows = [
        {
            "family": "control_dephasing",
            "parameter": 0.0,
            "generalized_robustness": 0.5,
            "status": "optimal",
            "witness_certificate_gap": 1e-10,
        },
        {
            "family": "control_dephasing",
            "parameter": 1.0,
            "generalized_robustness": 0.0,
            "status": "optimal",
            "witness_certificate_gap": 2e-10,
        },
        {
            "family": "white_visibility",
            "parameter": 0.0,
            "generalized_robustness": 0.0,
            "status": "optimal",
            "witness_certificate_gap": 1e-10,
        },
        {
            "family": "white_visibility",
            "parameter": 1.0,
            "generalized_robustness": 0.5,
            "status": "optimal",
            "witness_certificate_gap": 2e-10,
        },
    ]
    summary = landscape.summarize_rows(rows)
    assert summary["control_dephasing"]["monotone_nonincreasing"] is True
    assert summary["white_visibility"]["monotone_nondecreasing"] is True
    assert summary["white_visibility"]["zero_parameters"] == [0.0]


def test_landscape_write_outputs_strict_json_and_csv(tmp_path):
    rows = [
        {
            "family": "order_bias",
            "parameter": 0.5,
            "generalized_robustness": 0.545351,
            "status": "optimal",
            "solver": "SCS",
            "solver_version": "x",
            "cvxpy_version": "y",
            "num_iters": 1.0,
            "solve_time_s": 0.1,
            "equality_residual_fro": 1e-11,
            "subspace_residual_fro": 1e-11,
            "min_eig_W_AB": -1e-10,
            "min_eig_W_BA": -1e-10,
            "witness_value": 0.545351,
            "witness_certificate_gap": 1e-10,
        }
    ]
    json_path, csv_path = landscape.write_outputs(rows, tmp_path)
    text = json_path.read_text(encoding="utf-8")
    assert "NaN" not in text
    assert json.loads(text)["status"] == "switch_robustness_landscape"
    assert csv_path.read_text(encoding="utf-8").startswith("family,parameter,")


def test_landscape_main_uses_cli_grid_and_writes_outputs(tmp_path, monkeypatch):
    def fake_scan_family(family, values, solver="SCS", eps=1e-8, max_iters=200_000):
        return [
            {
                "family": family,
                "parameter": value,
                "generalized_robustness": value,
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
                "witness_value": value,
                "witness_certificate_gap": 1e-10,
            }
            for value in values
        ]

    monkeypatch.setattr(landscape, "scan_family", fake_scan_family)
    code = landscape.main(
        [
            "--families",
            "white_visibility,order_bias",
            "--grid",
            "0,1",
            "--outdir",
            str(tmp_path),
            "--solver",
            "FAKE",
        ]
    )
    assert code == 0
    report = json.loads((tmp_path / "switch_robustness_landscape.json").read_text(encoding="utf-8"))
    assert len(report["rows"]) == 4
    assert {row["family"] for row in report["rows"]} == {"white_visibility", "order_bias"}
