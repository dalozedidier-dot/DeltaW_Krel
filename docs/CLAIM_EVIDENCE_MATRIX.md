# Claim/evidence matrix for the DeltaW/K_rel manuscript

This file maps the manuscript claims to concrete repository artifacts.  Its
purpose is to make the supplement auditable: every strong statement in the
article should point either to executable evidence, a preregistered criterion,
or an explicit blocker.

## Evidence levels

| Level | Meaning | Appropriate manuscript language |
| --- | --- | --- |
| E0 | Conceptual / mathematical statement only | "We propose", "we define", "we motivate" |
| E1 | Toy or geometric simulation implemented | "proof of concept", "linearized control bench" |
| E2 | Process-matrix infrastructure implemented and tested | "projector/SDP infrastructure validation" |
| E3 | Ideal-switch benchmark implemented and reproduced | "submission-grade switch validation" |
| E4 | Experimental/tomographic validation on calibrated data | "experimental test" |

## Current claim status

| Manuscript claim | Current evidence | Repository artifact | Status |
| --- | --- | --- | --- |
| The admissible direction must be fixed before data. | E1 | `scripts/monte_carlo_control_supplement.py`, `config/config_preregistration.json` | Supported as a toy/geometric control rule. |
| The admissible space excludes calibrated noise and marginal/signalling directions. | E1 | `scripts/monte_carlo_control_supplement.py`, tests for orthogonality diagnostics | Supported in the linearized bench only. |
| The applicability lock prevents unstable normalization when the projected witness is too small. | E1 | `apply_applicability_lock`, `tests/test_monte_carlo_control_supplement.py` | Supported. |
| The finite-count Monte Carlo chain estimates sensitivity for lambda values. | E1 | `monte_carlo_outputs_control/`, `results/mc_smoke/` | Supported as proof of concept; not an experimental forecast. |
| A micro-tomography proof of concept is provided. | E1 | `scripts/micro_tomography_simulation.py`, `outputs/`, `results/micro_smoke/` | Supported as simplified upper-bound stress test. |
| Bipartite process-matrix trace/replace projectors are executable. | E2 | `src/deltawkrel/projectors.py`, `tests/test_projectors.py` | Supported, pending external convention audit. |
| The causal-SDP wiring over K_CS validates known causally separable targets. | E2 | `src/deltawkrel/sdp.py`, `scripts/run_sdp_validation.py`, CI `reproducibility-pipeline-outputs` | Supported for infrastructure validation. |
| The ideal quantum-switch benchmark is reproduced. | E3 | `src/deltawkrel/switch_models.py`, `src/deltawkrel/sdp.py`, `scripts/run_sdp_validation.py`, CI `sdp_validation_report.json` | Supported: generalized robustness reproduces the published value near 0.5454 within tolerance. |
| Partial control dephasing maps the falsification landscape between ideal and classical switch endpoints. | E3 | `partially_dephased_switch_process`, `scripts/run_switch_dephasing_scan.py` | Supported as a sampled SDP scan; continuum claims require an analytic proof or denser certified grid. |
| Robustness can be scanned across multiple switch perturbation families. | E3 | `biased_coherent_switch_process`, `white_visibility_switch_process`, `scripts/run_switch_robustness_landscape.py` | Supported as sampled SDP landscape over control dephasing, white-noise visibility, and coherent order bias. |
| The package is ready for a formal confirmatory submission. | E3 plus editorial cleanup | `docs/SUBMISSION_CHECKLIST.md` | Technically close; final submission still requires synchronized manuscript/notebook wording and archive DOI. |

## Rules for using this matrix

1. Claims labelled E1 must use cautious language: "toy", "linearized",
   "proof of concept", "upper-bound", or "smoke".
2. Claims labelled E2 can support software/infrastructure statements, not a
   completed physical benchmark.
3. E3 claims about the ideal switch must cite the implemented convention,
   benchmark tolerance, solver diagnostics, and the control-dephased negative
   control.
4. No claim should be promoted to E4 until calibrated experimental/tomographic
   data are available with the preregistered thresholds locked.

## Immediate upgrade targets

| Target | Evidence gain | Acceptance criterion |
| --- | --- | --- |
| Convention audit notebook | Stronger E3 | Equations, tensor order, trace convention, and normalization mapped line by line. |
| Partial-dephasing switch scan | Stronger E3 | JSON/CSV curve exported with monotonicity, endpoint, residual, and witness-gap diagnostics. |
| Multi-family switch landscape | Stronger E3 | Control-dephasing, white-visibility, and order-bias families exported with SDP diagnostics. |
| Solver diagnostics export | Required for E3 | Status, objective, dual gap if available, residuals, omega_white, complementarity checks. |
| Manifest validation command | Reproducibility hardening | `python scripts/validate_manifest.py` verifies that tracked artifacts match `MANIFEST.sha256.json`. |
| Zenodo release checklist | Publication hardening | DOI added to README/manuscript after archive. |
