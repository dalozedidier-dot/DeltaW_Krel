# DeltaW/K_rel reproducibility repository

This repository is a GitHub-ready implementation scaffold for the ΔW/K_rel manuscript. It contains the toy/geometric Monte Carlo bench, a micro-tomography proof-of-concept, explicit bipartite process-matrix projectors, a minimal K_CS causal-SDP validation block, tests, CI, and reproducibility templates.

## Current scientific status

### Implemented and testable

- Linearized Monte Carlo control script for Annex C bis.
- Micro-tomography proof-of-concept / upper-bound stress test.
- Explicit trace-and-replace maps for the `[AI, AO, BI, BO]` convention.
- Projectors `L_V`, `L_AB`, `L_BA` with idempotence tests.
- Exact validation targets:
  - white-noise process;
  - fixed-order `A≺B` identity-channel process;
  - fixed-order `B≺A` identity-channel process;
  - causally separable convex mixtures.
- Minimal SDP over `K_CS = C_AB + C_BA` validating zero robustness on causally separable targets when `cvxpy` is available.
- **Ideal quantum switch implemented** in the Araújo–Branciard–Costa–Feix–Giarmatzi–Brukner convention (NJP 17, 102001 (2015)): rank-one process on `[AI, AO, BI, BO, F]` with `F = F_target ⊗ F_control` (D = 64), plus the with-future projectors `L_{A≺B≺F}`, `L_{B≺A≺F}` and `L_V^F`.
- **Published benchmark reproduced**: the generalized robustness of the ideal switch computed by `solve_switch_generalized_robustness` is `0.545351` (SCS, eps 1e-8), matching the published `0.5454`; the control-dephased switch is causally separable (robustness 0). See `docs/CONVENTIONS.md` for the official convention and the equation-to-code audit table.
- Full SDP diagnostics exported (solver + versions, iterations, solve time, residuals, minimal eigenvalues, dual witness certificate) by `scripts/run_sdp_validation.py`.
- Unit/smoke tests and GitHub Actions CI (multi-Python coverage gate + reproducibility pipeline).

### Interpretation guardrails

The Monte Carlo control bench and the micro-tomography script are a
**methodological/geometric control level and a simplified stress test**. They
are useful and reproducible, but they must not be presented as experimental
validation or as physical simulations of the switch. Only the SDP level
carries the external published benchmark.


## Installation status

This package has been installed and smoke-tested in the sandbox using:

```bash
python -m pip install -e .
pytest -q
python scripts/run_sdp_validation.py
```

See `docs/INSTALL_AND_TEST_REPORT.md` for the exact validation commands and status. A local `.venv/` is intentionally **not** shipped in the ZIP, because virtual environments should not be committed to GitHub.

## Repository layout

```text
.github/workflows/ci.yml              GitHub Actions tests
src/deltawkrel/projectors.py          trace-replace maps and process projectors
src/deltawkrel/sdp.py                 minimal K_CS causal-SDP validation routine
src/deltawkrel/switch_models.py       white-noise/fixed-order targets + ideal quantum switch
scripts/monte_carlo_control_supplement.py
scripts/micro_tomography_simulation.py
scripts/run_sdp_validation.py
notebooks/projectors_definitions.ipynb
notebooks/validation_switch_ideal.ipynb
config/config_preregistration.json
tests/
docs/
```

## Quick start

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
python -m pip install --upgrade pip
python -m pip install -e .
pytest -q
```

Run the proof-of-concept scripts.  The smoke outputs are written under `results/` so the generated reproducibility report can find them without path translation:

```bash
python scripts/micro_tomography_simulation.py --outdir results/micro_smoke --n-sim 1000 --n-null 3000
python scripts/monte_carlo_control_supplement.py --n-sim 200 --n-null 500 --dim 4 --n-noise 1 --lambda-values 0,0.005 --n-values 1000 --output-dir results/mc_smoke --no-plots
```

Freeze the preregistration configuration and generate the reproducibility
report:

```bash
python scripts/freeze_preregistration.py
python scripts/generate_reproducibility_report.py
python scripts/validate_manifest.py
```

Run the SDP validation (bipartite targets + ideal-switch benchmark) after
installing `cvxpy` (solvers: SCS for the switch benchmark, CLARABEL for the
bipartite targets):

```bash
python scripts/run_sdp_validation.py
```

Full reproducibility from a clean clone:

```bash
make reproduce-full   # everything, requires cvxpy (SDP included)
make reproduce-core   # same chain without the SDP step (cvxpy-free environments)
```

Run the notebooks:

```bash
jupyter lab notebooks/projectors_definitions.ipynb
jupyter lab notebooks/validation_switch_ideal.ipynb
```

## Validation criteria before manuscript submission

The repository must be considered submission-ready only when all of the following are true:

1. `projectors_definitions.ipynb` exports `L_V`, `L_AB`, `L_BA` superoperators and documents the exact equations/conventions.
2. `validation_switch_ideal.ipynb` constructs the ideal quantum switch and reproduces a published robustness benchmark.
3. Solver diagnostics are exported: status, objective value, duality gap if available, `R_rel`, `omega_white`, and complementarity checks.
4. The final repository is archived with a DOI, for example via Zenodo.

For a claim-by-claim audit trail, see `docs/CLAIM_EVIDENCE_MATRIX.md`.
For the hardening roadmap, see `docs/MAXIMIZE_REPOSITORY_FOR_SUBMISSION.md`.

## Important methodological note

The scripts in this repository are useful for development and reproducibility, but the manuscript's confirmatory claim depends on the completed SDP/tomography validation. The micro-tomography and binomial simulations should be read as proof-of-concept / upper-bound stress tests, not full experimental predictions.
