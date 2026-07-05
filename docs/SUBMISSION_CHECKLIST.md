# Submission checklist

## Required before formal submission

- [ ] Confirm exact equations/conventions for `L_V`, `L_AB`, `L_BA`.
- [ ] Run `notebooks/projectors_definitions.ipynb` and export projector matrices.
- [x] Implement the ideal quantum-switch process matrix.
- [ ] Keep `notebooks/validation_switch_ideal.ipynb` synchronized with the published robustness benchmark.
- [x] Export SDP diagnostics: solver, status, objective, residuals, witness certificate, `omega_white`, and switch benchmark diagnostics.
- [ ] Freeze `config/config_preregistration.json` with `python scripts/freeze_preregistration.py`.
- [ ] Generate `results/reproducibility_report.md` in CI or release artifacts with `python scripts/generate_reproducibility_report.py`.
- [ ] Update `docs/CLAIM_EVIDENCE_MATRIX.md` so every manuscript claim has an artifact, status, or blocker.
- [ ] Complete the hardening steps in `docs/MAXIMIZE_REPOSITORY_FOR_SUBMISSION.md`.
- [ ] Run `pytest -q` locally and in GitHub Actions.
- [ ] Archive the final GitHub repository on Zenodo and include DOI in the manuscript.

## Already provided in this scaffold

- [x] Toy/geometric Monte Carlo bench.
- [x] Micro-tomography proof-of-concept.
- [x] Baseline trace-and-replace projectors.
- [x] Causal-SDP validation for CS/white-noise controls.
- [x] Ideal quantum-switch generalized-robustness benchmark.
- [x] CI workflow.
- [x] Pre-registration JSON template.
