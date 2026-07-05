# Submission checklist

## Required before formal submission

- [ ] Confirm exact equations/conventions for `L_V`, `L_AB`, `L_BA`.
- [ ] Run `notebooks/projectors_definitions.ipynb` and export projector matrices.
- [ ] Implement the ideal quantum-switch process matrix.
- [ ] Run `notebooks/validation_switch_ideal.ipynb` against a published robustness benchmark.
- [ ] Export SDP diagnostics: solver, status, objective, duality gap when available, `R_rel`, `omega_white`, complementarity checks.
- [ ] Update `docs/CLAIM_EVIDENCE_MATRIX.md` so every manuscript claim has an artifact, status, or blocker.
- [ ] Complete the hardening steps in `docs/MAXIMIZE_REPOSITORY_FOR_SUBMISSION.md`.
- [ ] Run `pytest -q` locally and in GitHub Actions.
- [ ] Archive the final GitHub repository on Zenodo and include DOI in the manuscript.

## Already provided in this scaffold

- [x] Toy/geometric Monte Carlo bench.
- [x] Micro-tomography proof-of-concept.
- [x] Baseline trace-and-replace projectors.
- [x] Minimal causal-SDP scaffold for safe CS/white-noise validation.
- [x] CI workflow.
- [x] Pre-registration JSON template.
