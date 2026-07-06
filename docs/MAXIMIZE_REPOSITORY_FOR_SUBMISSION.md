# How to push this repository to submission strength

The repository is already useful: it contains executable toy simulations,
process-matrix projectors, a minimal causal-SDP scaffold, smoke outputs, tests,
CI, preregistration parameters, and a SHA manifest.  The next step is not to add
volume.  The next step is to turn the repository into an auditable evidence
package for the manuscript.

## Priority 1: keep the switch benchmark auditable

The main scientific benchmark is now implemented: `ideal_quantum_switch_process()`
constructs the ideal switch in the Araújo et al. convention, and
`scripts/run_sdp_validation.py` reproduces the published generalized robustness
near `0.5454` within the declared tolerance.

To keep this submission-grade:

1. Keep the convention visible in code, docs, and notebook.
2. Keep tests for shape, Hermiticity, PSD, trace convention, projectors, and
   benchmark tolerance.
3. Keep the control-dephased switch as a negative control with robustness near
   zero.
4. Export complete diagnostics through the CI pipeline artifacts.
5. Avoid committing stale generated reports that refer to absent artifacts.
6. Use `scripts/run_switch_robustness_landscape.py` for the stronger
   multi-family SDP landscape: control dephasing, white-noise visibility, and
   coherent order bias.
7. Use `scripts/realistic_tomography_pipeline.py` for the heavier finite-count
   simulated tomography bridge: multinomial/Poisson counts, reconstruction,
   covariance shrinkage, bootstrap, and power maps.

## Priority 2: make the manuscript auditable

Use `docs/CLAIM_EVIDENCE_MATRIX.md` as the spine of the supplement.  For every
major claim in the article, keep one of three states:

| State | Meaning |
| --- | --- |
| Supported | The repo has an executable artifact and test/output. |
| Partially supported | The repo has a toy/scaffold version only. |
| Blocked | The repo intentionally refuses to claim completion. |

This protects the work from overclaiming and makes peer review easier.

## Priority 3: harden reproducibility

Recommended additions:

1. Keep the preregistration lock green: `python scripts/freeze_preregistration.py`.
2. Keep the manifest validation command green: `python scripts/validate_manifest.py`.
3. Keep `make reproduce-full` green; it runs preregistration lock, tests, SDP,
   micro-tomography smoke, Monte Carlo smoke, and manifest validation.
4. Store generated smoke outputs under the canonical directories `results/micro_smoke/` and `results/mc_smoke/` with deterministic seeds.
5. Keep `CITATION.cff` ready for the archive.
6. Archive a release through Zenodo and add the DOI to the README and article.
7. Keep a light realistic-tomography smoke run green in CI, with heavier power
   maps generated as release artifacts rather than committed source files.

## Priority 4: improve numerical credibility

Before submission, the SDP layer should report more than "optimal":

| Diagnostic | Why it matters |
| --- | --- |
| primal residual | verifies equality constraints |
| dual residual / dual gap | verifies numerical optimality when available |
| omega_white | detects domination by theoretical white noise |
| complementarity check | validates primal/dual consistency |
| solver/version/tolerance | makes results reproducible |
| convention checksum | prevents silent tensor-order drift |

If CLARABEL is the open-source default, an optional MOSEK cross-check can be
reported separately when available.

## Priority 5: align wording in the article

The article can safely say:

- the DeltaW/K_rel decision architecture is preregistered at the level of a
  computational protocol;
- the toy/geometric Monte Carlo bench validates the statistical logic;
- the micro-tomography script is a simplified proof of concept;
- the process-matrix and K_CS SDP infrastructure validate known causally
  separable targets;
- the ideal quantum-switch benchmark is reproduced against the published
  generalized robustness reference near 0.5454.
- sampled SDP robustness landscapes are available for control dephasing,
  white-noise visibility, and coherent order-bias perturbations.
- finite-count simulated tomography now supports counts, reconstruction,
  covariance shrinkage, bootstrap, and power maps under visibility/crosstalk/drift.

The article should not say:

- the reported Monte Carlo power is an experimental prediction;
- the micro-tomography stress test is an experimental validation;
- the repository has a final archival DOI until Zenodo or an equivalent archive
  has minted one.

## Suggested final repository shape

```text
README.md
CITATION.cff
environment.yml
pyproject.toml
config/
  config_preregistration.json
docs/
  CLAIM_EVIDENCE_MATRIX.md
  MAXIMIZE_REPOSITORY_FOR_SUBMISSION.md
  REPRODUCIBILITY_NOTES.md
  SDP_VALIDATION_STATUS.md
  SUBMISSION_CHECKLIST.md
notebooks/
  projectors_definitions.ipynb
  validation_switch_ideal.ipynb
results/
  sdp_validation_report.json
  mc_smoke/        canonical Monte Carlo smoke outputs
  micro_smoke/     canonical micro-tomography smoke outputs
scripts/
src/deltawkrel/
tests/
```

## Decision rule

The repository is "maxed out" for submission only when a reviewer can start
from a clean clone, run one documented command, and see:

1. all tests pass;
2. toy and micro-tomography smoke outputs regenerate;
3. the realistic finite-count tomography bridge regenerates a smoke power map;
4. the SDP validates known CS targets;
5. the ideal quantum switch benchmark is reproduced and the dephased-switch
   negative control remains near zero;
6. every manuscript claim maps to an artifact in
   `docs/CLAIM_EVIDENCE_MATRIX.md`.
