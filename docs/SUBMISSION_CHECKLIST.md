# Submission checklist

## Required before formal submission

- [x] Confirm exact equations/conventions for `L_V`, `L_AB`, `L_BA` in
  `docs/CONVENTIONS.md` and regression tests.
- [ ] Run `notebooks/projectors_definitions.ipynb` and export projector matrices.
- [x] Implement the ideal quantum-switch process matrix.
- [x] Keep `notebooks/validation_switch_ideal.ipynb` synchronized with the published robustness benchmark.
- [x] Export SDP diagnostics: solver, status, objective, residuals, witness certificate, `omega_white`, and switch benchmark diagnostics.
- [x] Freeze `config/config_preregistration.json` with `python scripts/freeze_preregistration.py`.
- [x] Generate `results/reproducibility_report.md` locally with `python scripts/generate_reproducibility_report.py`.
- [x] Add realistic finite-count tomography simulation bridge with counts, reconstruction, covariance shrinkage, bootstrap, and power maps.
- [x] Add full simulated tomography stress test with path losses, dephasing, crosstalk, drift, interleaved controls, LR calibration, and applicability map.
- [x] Add A2 certificate lemmas, A4 finite-count scalar statistics, and A5 certified robustness interval.
- [x] Update `docs/CLAIM_EVIDENCE_MATRIX.md` so every current repository claim has an artifact, status, or blocker.
- [x] Run the full local test suite with coverage (`197 passed`, `96.45%`).
- [ ] Confirm the latest GitHub Actions run is green after deployment.
- [ ] Generate `results/reproducibility_report.md` as a release/CI artifact, not only locally.
- [ ] Complete the remaining hardening steps in `docs/MAXIMIZE_REPOSITORY_FOR_SUBMISSION.md`.
- [ ] Archive the final GitHub repository on Zenodo and include DOI in the manuscript.
- [ ] Ingest or explicitly defer public experimental datasets; current status is inventory only, no raw external-data reanalysis.

## Already provided in this scaffold

- [x] Toy/geometric Monte Carlo bench.
- [x] Micro-tomography proof-of-concept.
- [x] Realistic finite-count tomography simulation bridge.
- [x] Full simulated tomography stress test.
- [x] A2 certificate lemmas.
- [x] A4 finite-count single-direction estimator.
- [x] A5 certified primal/dual robustness interval.
- [x] Public-data inventory and positioning document.
- [x] Baseline trace-and-replace projectors.
- [x] Causal-SDP validation for CS/white-noise controls.
- [x] Ideal quantum-switch generalized-robustness benchmark.
- [x] CI workflow.
- [x] Pre-registration JSON template.
