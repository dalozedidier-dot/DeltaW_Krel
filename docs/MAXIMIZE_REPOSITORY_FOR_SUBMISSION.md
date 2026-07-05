# How to push this repository to submission strength

The repository is already useful: it contains executable toy simulations,
process-matrix projectors, a minimal causal-SDP scaffold, smoke outputs, tests,
CI, preregistration parameters, and a SHA manifest.  The next step is not to add
volume.  The next step is to turn the repository into an auditable evidence
package for the manuscript.

## Priority 1: close the scientific blocker

The main blocker is explicit and healthy: `ideal_quantum_switch_process()` is
not implemented by design.

To make the repository submission-grade:

1. Select one published ideal quantum-switch convention.
2. Document the Hilbert-space order, trace convention, normalization, and
   future/control system.
3. Implement the process in `src/deltawkrel/switch_models.py`.
4. Add tests for shape, Hermiticity, PSD, trace convention, and projector
   compatibility.
5. Run the causal-SDP routine on the ideal switch.
6. Reproduce a published robustness value within a declared numerical tolerance.
7. Export the complete diagnostics to `results/sdp_validation_report.json`.

Until this is done, the manuscript should keep the current wording: proof of
concept / infrastructure validation / roadmap, not completed quantum-switch
benchmark.

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

1. Keep the manifest validation command green: `python scripts/validate_manifest.py`.
2. Keep a single `make reproduce` target that runs tests,
   micro-tomography smoke, Monte Carlo smoke, and manifest validation.
3. Store generated smoke outputs under `results/` with deterministic seeds.
4. Add a short `CITATION.cff`.
5. Archive a release through Zenodo and add the DOI to the README and article.

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
- the ideal quantum-switch benchmark remains the decisive completion step.

The article should not yet say:

- the ideal quantum switch has been reproduced;
- the reported Monte Carlo power is an experimental prediction;
- the current SDP proves non-separability of an implemented switch process;
- the repository is a final confirmatory supplement.

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
  mc_smoke/
  micro_smoke/
scripts/
src/deltawkrel/
tests/
```

## Decision rule

The repository is "maxed out" for submission only when a reviewer can start
from a clean clone, run one documented command, and see:

1. all tests pass;
2. toy and micro-tomography smoke outputs regenerate;
3. the SDP validates known CS targets;
4. the ideal quantum switch benchmark is reproduced or explicitly marked as
   absent;
5. every manuscript claim maps to an artifact in
   `docs/CLAIM_EVIDENCE_MATRIX.md`.
