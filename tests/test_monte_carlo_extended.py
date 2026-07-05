"""Extended unit and integration tests for the linearized ΔW/K_rel control bench.

Covers the algebraic helpers, the M0 marginal constraints, the statistical
layer (Wilson intervals, LR thresholds, p-values), the applicability lock, the
witness-stability diagnostic, the regularized SDP proxy, exports/plots and the
command-line entry point. Everything remains at the toy/geometric level.
"""
from __future__ import annotations

import argparse
import json

import numpy as np
import pytest
from scipy.stats import chi2

import monte_carlo_control_supplement as mc


# ------------------------------------------------------------------
# Algebraic helpers
# ------------------------------------------------------------------


def test_symmetrize_and_traceless_batched():
    rng = np.random.default_rng(0)
    batch = rng.normal(size=(7, 4, 4))
    sym = mc.symmetrize(batch)
    assert np.allclose(sym, np.swapaxes(sym, -1, -2))
    tl = mc.traceless(sym)
    assert np.allclose(np.trace(tl, axis1=-2, axis2=-1), 0.0, atol=1e-12)


def test_normalize_frobenius():
    M = 3.0 * np.eye(2)
    out = mc.normalize_frobenius(M)
    assert np.isclose(np.linalg.norm(out), 1.0)
    with pytest.raises(ValueError):
        mc.normalize_frobenius(np.zeros((3, 3)))


def test_frob_inner_matches_trace_formula():
    rng = np.random.default_rng(1)
    A = rng.normal(size=(5, 5))
    B = rng.normal(size=(5, 5))
    assert np.isclose(mc.frob_inner(A, B), float(np.trace(A.T @ B)))


def test_make_symmetric_basis_is_orthonormal_and_complete():
    dim = 4
    basis = mc.make_symmetric_basis(dim)
    assert len(basis) == dim * (dim + 1) // 2
    gram = np.array([[mc.frob_inner(a, b) for b in basis] for a in basis])
    assert np.allclose(gram, np.eye(len(basis)), atol=1e-12)


def test_project_onto_basis_recovers_spanned_matrix():
    basis = mc.make_symmetric_basis(3)
    rng = np.random.default_rng(2)
    M = mc.symmetrize(rng.normal(size=(3, 3)))
    assert np.allclose(mc.project_onto_basis(M, basis), M, atol=1e-12)


def test_gram_schmidt_add_rejects_dependent_directions():
    basis = [np.eye(2) / np.sqrt(2.0)]
    assert mc.gram_schmidt_add(5.0 * np.eye(2), basis) is None
    new = mc.gram_schmidt_add(np.diag([1.0, -1.0]), basis)
    assert new is not None
    assert np.isclose(np.linalg.norm(new), 1.0)
    assert abs(mc.frob_inner(new, basis[0])) < 1e-12


def test_random_traceless_symmetric_invariants():
    rng = np.random.default_rng(3)
    M = mc.random_traceless_symmetric(6, rng)
    assert np.allclose(M, M.T)
    assert abs(np.trace(M)) < 1e-10
    assert np.isclose(np.linalg.norm(M), 1.0)


# ------------------------------------------------------------------
# Partial traces and M0 marginal constraints
# ------------------------------------------------------------------


def test_partial_traces_on_kron_product():
    rng = np.random.default_rng(4)
    A = rng.normal(size=(2, 2))
    B = rng.normal(size=(3, 3))
    M = np.kron(A, B)
    assert np.allclose(mc.partial_trace_A(M, 2, 3), np.trace(A) * B, atol=1e-12)
    assert np.allclose(mc.partial_trace_B(M, 2, 3), np.trace(B) * A, atol=1e-12)


def test_marginal_constraint_matrices_counts_and_errors():
    constraints = mc.marginal_constraint_matrices(4, subsystem_dim=2)
    assert len(constraints) == 2 * 2 + 2 * 2  # dB^2 + dA^2
    for C in constraints:
        assert np.allclose(C, C.T)
    # Automatic factorization for composite dimension.
    auto = mc.marginal_constraint_matrices(6)
    assert len(auto) == 3 * 3 + 2 * 2
    with pytest.raises(ValueError):
        mc.marginal_constraint_matrices(7)  # prime dimension
    with pytest.raises(ValueError):
        mc.marginal_constraint_matrices(6, subsystem_dim=4)  # does not divide


def test_build_admissible_space_dimension_counting():
    dim = 4
    rng = np.random.default_rng(5)
    noise = mc.build_noise_basis(dim=dim, n_noise=2, rng=rng)
    A_basis = mc.build_admissible_space(dim=dim, N_cal_basis=noise)
    # sym space (10) minus trace (1) minus 2 independent traceless noises.
    assert len(A_basis) == dim * (dim + 1) // 2 - 1 - 2
    for M in A_basis:
        assert abs(np.trace(M)) < 1e-8
        for N in noise:
            assert abs(mc.frob_inner(M, N)) < 1e-8


def test_build_admissible_space_without_constraints_is_full_symmetric_space():
    A_basis = mc.build_admissible_space(dim=3, N_cal_basis=[], impose_trace_zero=False)
    assert len(A_basis) == 3 * 4 // 2


def test_build_admissible_space_with_M0_kills_marginals():
    dim = 4
    rng = np.random.default_rng(6)
    noise = mc.build_noise_basis(dim=dim, n_noise=1, rng=rng)
    A_basis = mc.build_admissible_space(
        dim=dim, N_cal_basis=noise, include_M0=True, subsystem_dim=2
    )
    assert len(A_basis) > 0
    for M in A_basis:
        assert np.allclose(mc.partial_trace_A(M, 2, 2), 0.0, atol=1e-8)
        assert np.allclose(mc.partial_trace_B(M, 2, 2), 0.0, atol=1e-8)


# ------------------------------------------------------------------
# Statistical layer
# ------------------------------------------------------------------


def test_wilson_ci_properties():
    assert all(np.isnan(v) for v in mc.wilson_ci(0, 0))
    low, high = mc.wilson_ci(5, 10)
    assert 0.0 <= low < 0.5 < high <= 1.0
    low0, high0 = mc.wilson_ci(0, 20)
    assert low0 == 0.0 and high0 > 0.0
    low1, high1 = mc.wilson_ci(20, 20)
    assert high1 == 1.0 and low1 < 1.0


def test_asymptotic_lr_threshold_values_and_errors():
    one_sided = mc.asymptotic_lr_threshold(0.01, lambda_positive_only=True)
    two_sided = mc.asymptotic_lr_threshold(0.01, lambda_positive_only=False)
    assert np.isclose(one_sided, chi2.isf(0.02, df=1))
    assert np.isclose(two_sided, chi2.isf(0.01, df=1))
    assert one_sided < two_sided
    for bad in (0.0, 1.0, -0.1):
        with pytest.raises(ValueError):
            mc.asymptotic_lr_threshold(bad, lambda_positive_only=False)
    with pytest.raises(ValueError):
        mc.asymptotic_lr_threshold(0.6, lambda_positive_only=True)


def test_asymptotic_p_values_boundary_mixture():
    LR = np.array([0.0, 4.0])
    one_sided = mc.asymptotic_p_values(LR, lambda_positive_only=True)
    two_sided = mc.asymptotic_p_values(LR, lambda_positive_only=False)
    assert one_sided[0] == 1.0
    assert np.isclose(one_sided[1], 0.5 * chi2.sf(4.0, df=1))
    assert np.isclose(two_sided[1], chi2.sf(4.0, df=1))


def test_empirical_p_values_plus_one_correction():
    null_lr = np.array([1.0, 2.0, 3.0, 4.0])
    p = mc.empirical_p_values(np.array([5.0, 2.0, 0.0]), null_lr)
    assert np.allclose(p, [(0 + 1) / 5, (3 + 1) / 5, (4 + 1) / 5])
    with pytest.raises(ValueError):
        mc.empirical_p_values(np.array([1.0]), np.array([]))


# ------------------------------------------------------------------
# Noise basis and applicability lock
# ------------------------------------------------------------------


def test_build_noise_basis_orthonormal_and_bounds():
    rng = np.random.default_rng(7)
    basis = mc.build_noise_basis(dim=4, n_noise=3, rng=rng)
    for i, N in enumerate(basis):
        assert np.isclose(np.linalg.norm(N), 1.0)
        assert abs(np.trace(N)) < 1e-10
        for j in range(i):
            assert abs(mc.frob_inner(N, basis[j])) < 1e-10
    with pytest.raises(ValueError):
        mc.build_noise_basis(dim=4, n_noise=0, rng=rng)
    with pytest.raises(ValueError):
        mc.build_noise_basis(dim=2, n_noise=2, rng=rng)


def test_basis_diagnostics_reports_expected_keys():
    rng = np.random.default_rng(8)
    basis = mc.build_basis_control(dim=4, rng=rng, alpha_true=0.12, n_noise=2)
    diag = mc.basis_diagnostics(basis)
    assert np.isclose(diag["trace_W_QM"], 1.0)
    assert np.isclose(diag["norm_K_rel"], 1.0)
    assert diag["applicable"] == 1.0
    assert 0.0 < diag["proj_norm_relative_to_random"]
    assert "inner_N_cal_1_0" in diag
    assert diag["max_inner_N_K_rel"] < 1e-8


# ------------------------------------------------------------------
# Witness-stability diagnostic
# ------------------------------------------------------------------


def _small_basis(seed: int = 9) -> mc.Basis:
    rng = np.random.default_rng(seed)
    return mc.build_basis_control(dim=4, rng=rng, alpha_true=0.12, n_noise=1)


def test_witness_stability_zero_samples_branch():
    out = mc.witness_stability_diagnostics(_small_basis(), np.random.default_rng(0), n_samples=0)
    assert out["witness_stability_run"] == 0.0
    assert np.isnan(out["witness_cos_min"])


def test_witness_stability_nominal_run():
    out = mc.witness_stability_diagnostics(
        _small_basis(), np.random.default_rng(1), n_samples=50, perturb_scale=0.01
    )
    assert out["witness_stability_run"] == 1.0
    assert out["n_witness_samples"] == 50.0
    assert 0.0 <= out["witness_cos_min"] <= out["witness_cos_q50"] <= 1.0 + 1e-12
    # Tiny perturbations must leave the direction essentially unchanged.
    assert out["witness_cos_mean"] > 0.99
    assert out["witness_unstable_alert"] == 0.0


def test_witness_stability_all_rejected_branch():
    out = mc.witness_stability_diagnostics(
        _small_basis(), np.random.default_rng(2), n_samples=10, eps_num=1e3
    )
    assert out["witness_rejected_samples"] == 10.0
    assert out["witness_unstable_alert"] == 1.0
    assert np.isnan(out["witness_cos_mean"])


# ------------------------------------------------------------------
# Simulation and estimation
# ------------------------------------------------------------------


def test_simulate_observations_validates_inputs_and_recenters_trace():
    W = np.eye(4) / 4.0
    rng = np.random.default_rng(10)
    with pytest.raises(ValueError):
        mc.simulate_observations(W, n_samples=0, n_sim=2, sigma_obs=0.1, rng=rng)
    with pytest.raises(ValueError):
        mc.simulate_observations(W, n_samples=10, n_sim=0, sigma_obs=0.1, rng=rng)
    with pytest.raises(ValueError):
        mc.simulate_observations(W, n_samples=10, n_sim=2, sigma_obs=0.0, rng=rng)
    W_obs = mc.simulate_observations(W, n_samples=100, n_sim=32, sigma_obs=0.1, rng=rng)
    assert W_obs.shape == (32, 4, 4)
    assert np.allclose(np.trace(W_obs, axis1=-2, axis2=-1), 1.0, atol=1e-10)
    assert np.allclose(W_obs, np.swapaxes(W_obs, -1, -2), atol=1e-12)


def test_estimate_lr_vectorized_shapes_and_positivity():
    basis = _small_basis()
    rng = np.random.default_rng(11)
    W_obs = mc.simulate_observations(basis.W_0, n_samples=50, n_sim=16, sigma_obs=0.05, rng=rng)
    with pytest.raises(ValueError):
        mc.estimate_lr_vectorized(W_obs[0], basis, n_samples=50, sigma_obs=0.05)
    est = mc.estimate_lr_vectorized(W_obs, basis, n_samples=50, sigma_obs=0.05)
    assert est["beta_hats"].shape == (16, len(basis.N_cal_basis))
    assert np.all(est["lambda_hat"] >= 0.0)
    assert np.all(est["LR"] >= 0.0)
    est2 = mc.estimate_lr_vectorized(
        W_obs, basis, n_samples=50, sigma_obs=0.05, lambda_positive_only=False
    )
    assert np.allclose(est2["lambda_hat"], est2["lambda_raw"])
    assert np.any(est2["lambda_raw"] < 0.0)  # two-sided keeps negatives


def test_estimate_recovers_injected_lambda_without_bias():
    basis = _small_basis()
    rng = np.random.default_rng(12)
    lam = 0.05
    W_true = basis.W_0 + lam * basis.K_rel
    W_obs = mc.simulate_observations(W_true, n_samples=5000, n_sim=200, sigma_obs=0.05, rng=rng)
    est = mc.estimate_lr_vectorized(W_obs, basis, n_samples=5000, sigma_obs=0.05)
    assert np.mean(est["lambda_hat"]) == pytest.approx(lam, abs=5e-3)


def test_calibrate_thresholds_modes_and_errors():
    basis = _small_basis()
    rng = np.random.default_rng(13)
    with pytest.raises(ValueError):
        mc.calibrate_thresholds(
            basis, [100], sigma_obs=0.05, alpha_level=0.01,
            lambda_positive_only=True, threshold_mode="bogus", n_null=10, rng=rng,
        )
    asym = mc.calibrate_thresholds(
        basis, [100], sigma_obs=0.05, alpha_level=0.01,
        lambda_positive_only=True, threshold_mode="asymptotic", n_null=10, rng=rng,
    )
    assert asym[100].null_lr is None
    assert np.isclose(asym[100].threshold, chi2.isf(0.02, df=1))
    emp = mc.calibrate_thresholds(
        basis, [100], sigma_obs=0.05, alpha_level=0.05,
        lambda_positive_only=True, threshold_mode="empirical", n_null=500, rng=rng,
    )
    assert emp[100].null_lr is not None
    assert len(emp[100].null_lr) == 500
    assert emp[100].threshold >= 0.0


# ------------------------------------------------------------------
# SDP proxy (requires cvxpy)
# ------------------------------------------------------------------


def test_not_run_diagnostics_defaults():
    diag = mc.not_run_robustness_diagnostics()
    assert diag.status == "not_run"
    assert np.isnan(diag.omega_white)
    assert diag.omega_white_alert is False


def test_make_white_noise_process():
    W = mc.make_white_noise_process(5)
    assert np.isclose(np.trace(W), 1.0)
    assert np.allclose(W, np.eye(5) / 5.0)


def test_sdp_proxy_requires_cvxpy(monkeypatch):
    monkeypatch.setattr(mc, "cp", None)
    with pytest.raises(RuntimeError):
        mc.solve_regularized_sdp_proxy(np.eye(2), [])


def test_sdp_proxy_trivially_feasible_psd_target():
    pytest.importorskip("cvxpy")
    diag = mc.solve_regularized_sdp_proxy(np.eye(3), [np.diag([1.0, -1.0, 0.0])])
    assert diag.status in {"optimal", "optimal_inaccurate"}
    assert diag.objective_value == pytest.approx(0.0, abs=1e-6)
    assert diag.omega_white_alert is False


def test_sdp_proxy_known_deficit_is_repaired_by_white_noise():
    pytest.importorskip("cvxpy")
    # W = diag(-0.3, 1) with white noise I/2: needs t_white/2 >= 0.3.
    W = np.diag([-0.3, 1.0])
    diag = mc.solve_regularized_sdp_proxy(W, [])
    assert diag.status in {"optimal", "optimal_inaccurate"}
    assert diag.objective_value == pytest.approx(0.6, abs=1e-5)
    assert diag.omega_white == pytest.approx(1.0, abs=1e-6)
    assert diag.omega_white_alert is True


# ------------------------------------------------------------------
# End-to-end Monte Carlo runs
# ------------------------------------------------------------------


def test_run_is_reproducible_given_seeds():
    kwargs = dict(
        lambda_true_values=[0.0, 0.01],
        n_samples_values=[100],
        n_sim=20,
        dim=4,
        n_noise=1,
        seed_basis=21,
        seed_sim=22,
        save_outputs=False,
        make_plots=False,
    )
    rows_a = mc.run_monte_carlo_control(**kwargs)
    rows_b = mc.run_monte_carlo_control(**kwargs)
    assert [r.power for r in rows_a] == [r.power for r in rows_b]
    assert [r.lambda_fit_mean for r in rows_a] == [r.lambda_fit_mean for r in rows_b]


def test_power_grows_with_signal_and_sample_size():
    rows = mc.run_monte_carlo_control(
        lambda_true_values=[0.0, 0.05],
        n_samples_values=[100, 5000],
        n_sim=100,
        dim=4,
        n_noise=1,
        sigma_obs=0.05,
        seed_basis=31,
        seed_sim=32,
        save_outputs=False,
        make_plots=False,
    )
    by_key = {(r.lambda_true, r.n_samples): r.power for r in rows}
    # Type-I error stays near alpha at the null.
    assert by_key[(0.0, 5000)] <= 0.10
    # Strong signal at large N must be detected essentially always.
    assert by_key[(0.05, 5000)] >= 0.95
    assert by_key[(0.05, 5000)] >= by_key[(0.05, 100)]


def test_inapplicable_configuration_returns_empty(capsys):
    rows = mc.run_monte_carlo_control(
        lambda_true_values=[0.0],
        n_samples_values=[100],
        n_sim=5,
        dim=4,
        n_noise=1,
        eps_num=1e3,
        save_outputs=False,
        make_plots=False,
    )
    assert rows == []
    assert "TEST INAPPLICABLE" in capsys.readouterr().out


def test_full_run_exports_all_artifacts(tmp_path):
    pytest.importorskip("cvxpy")
    outdir = tmp_path / "mc_outputs"
    rows = mc.run_monte_carlo_control(
        lambda_true_values=[0.0, 0.005],
        n_samples_values=[100],
        n_sim=15,
        n_null=60,
        dim=4,
        n_noise=1,
        seed_basis=41,
        seed_sim=42,
        threshold_mode="empirical",
        include_M0=True,
        subsystem_dim=2,
        output_dir=str(outdir),
        save_outputs=True,
        make_plots=True,
        sdp_check=True,
        witness_stability=True,
        n_witness_samples=10,
    )
    assert len(rows) == 2
    for name in (
        "monte_carlo_power_results.csv",
        "monte_carlo_power_results.json",
        "basis_diagnostics.json",
        "simulation_config.json",
        "sdp_psd_proxy_diagnostics.json",
        "witness_stability_diagnostics.json",
        "power_curves.png",
        "power_heatmap.png",
    ):
        assert (outdir / name).exists(), f"missing artifact: {name}"
    payload = json.loads((outdir / "monte_carlo_power_results.json").read_text())
    assert len(payload) == 2
    assert payload[0]["include_M0"] is True
    config = json.loads((outdir / "simulation_config.json").read_text())
    assert config["seed_basis"] == 41
    assert config["seed_sim"] == 42


def test_export_results_rejects_empty(tmp_path):
    basis = _small_basis()
    with pytest.raises(ValueError):
        mc.export_results(results=[], basis=basis, output_path=tmp_path, config={})


# ------------------------------------------------------------------
# CLI layer
# ------------------------------------------------------------------


def test_parse_float_and_int_lists():
    assert mc.parse_float_list("0, 0.5 ,1") == [0.0, 0.5, 1.0]
    assert mc.parse_int_list("10,20") == [10, 20]
    with pytest.raises(argparse.ArgumentTypeError):
        mc.parse_float_list(" , ")
    with pytest.raises(argparse.ArgumentTypeError):
        mc.parse_int_list("")
    with pytest.raises(argparse.ArgumentTypeError):
        mc.parse_int_list("100,0")


def test_main_cli_smoke(tmp_path, monkeypatch):
    outdir = tmp_path / "cli_outputs"
    monkeypatch.setattr(
        "sys.argv",
        [
            "monte_carlo_control_supplement.py",
            "--n-sim", "10",
            "--n-null", "30",
            "--dim", "4",
            "--n-noise", "1",
            "--seed-basis", "51",
            "--seed-sim", "52",
            "--lambda-values", "0,0.005",
            "--n-values", "100",
            "--threshold-mode", "empirical",
            "--two-sided",
            "--witness-stability",
            "--n-witness-samples", "5",
            "--no-plots",
            "--output-dir", str(outdir),
        ],
    )
    mc.main()
    assert (outdir / "monte_carlo_power_results.csv").exists()
    config = json.loads((outdir / "simulation_config.json").read_text())
    assert config["lambda_positive_only"] is False
    assert config["threshold_mode"] == "empirical"
