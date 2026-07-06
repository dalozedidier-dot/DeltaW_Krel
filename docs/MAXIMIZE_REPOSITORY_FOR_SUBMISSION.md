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
7. Use `scripts/run_certified_witness_landscape.py` for the E3+ fixed-witness
   landscape: one dual-optimal witness extracted at the ideal switch, reused
   across the three switch perturbation families with full-SDP lower-bound
   checks.
8. Use `scripts/run_certified_bounds.py` for the A5 certified interval:
   primal/dual robustness bounds, solver status, and numerical bracket width.
9. Use `scripts/run_finite_count_analysis.py` for A4: one-scalar shot-noise
   scaling, copies-to-certify surfaces, and calibrated-drift false-positive
   contrast between `K_rel` and the raw witness.
10. Use `scripts/realistic_tomography_pipeline.py` for the heavier finite-count
   simulated tomography bridge: multinomial/Poisson counts, reconstruction,
   covariance shrinkage, bootstrap, and power maps.
11. Use `scripts/full_tomography_simulation.py` for the state-of-the-art
   simulated stress test: path-dependent loss, dephasing, crosstalk, temporal
   drift, interleaved controls, empirical LR calibration, and applicability map.

## Priority 2: make the manuscript auditable

Use `docs/CLAIM_EVIDENCE_MATRIX.md` as the spine of the supplement.  For every
major claim in the article, keep one of three states:

| State | Meaning |
| --- | --- |
| Supported | The repo has an executable artifact and test/output. |
| Partially supported | The repo has a toy/scaffold version only. |
| Blocked | The repo intentionally refuses to claim completion. |

This protects the work from overclaiming and makes peer review easier.

For the longer research program beyond the submission package, keep
`docs/ULTIMATE_VISION_ROADMAP.md` synchronized with the claim/evidence matrix.
That roadmap is an ambition tracker; it must not promote future work to current
evidence.

## Priority 3: harden reproducibility

Recommended additions:

1. Keep the preregistration lock green: `python scripts/freeze_preregistration.py`.
2. Keep the manifest validation command green: `python scripts/validate_manifest.py`.
3. Keep `make reproduce-full` green; it runs preregistration lock, tests, SDP,
   certified witness/bounds, finite-count statistics, micro-tomography smoke,
   Monte Carlo smoke, and manifest validation.
4. Store generated smoke outputs under the canonical directories `results/micro_smoke/` and `results/mc_smoke/` with deterministic seeds.
5. Keep `CITATION.cff` ready for the archive.
6. Archive a release through Zenodo and add the DOI to the README and article.
7. Keep a light realistic-tomography smoke run green in CI, with heavier power
   maps generated as release artifacts rather than committed source files.
8. Keep a light full-tomography stress smoke run green in CI; reserve massive
   10k-50k bootstrap/null runs for scheduled or release jobs.

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
- a fixed certified dual witness lower-bounds SDP robustness across the three
  switch perturbation families, with explicit certified regions and non-E4
  guardrails.
- the ideal-switch robustness is bracketed by a primal/dual certified interval,
  not only a solver `optimal` status.
- the fixed witness admits a finite-count one-scalar estimator with 1/N
  shot-noise scaling and calibrated-drift false-positive controls.
- finite-count simulated tomography now supports counts, reconstruction,
  covariance shrinkage, bootstrap, and power maps under visibility/crosstalk/drift.
- full simulated tomography stress tests now include path losses, dephasing,
  crosstalk, temporal drift, interleaved controls, empirical LR calibration,
  false-positive drift checks, and applicability maps.

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
  CERTIFICATE_LEMMAS.md
  CLAIM_EVIDENCE_MATRIX.md
  MAXIMIZE_REPOSITORY_FOR_SUBMISSION.md
  PROTOCOL_POSITIONING_AND_DATA_INVENTORY.md
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
4. the full tomography stress test regenerates its report and applicability map;
5. the SDP validates known CS targets;
6. the ideal quantum switch benchmark is reproduced and the dephased-switch
   negative control remains near zero;
7. the fixed-witness layer exports A2 lemmas, A4 finite-count statistics, and
   A5 certified intervals;
8. every manuscript claim maps to an artifact in
   `docs/CLAIM_EVIDENCE_MATRIX.md`.
