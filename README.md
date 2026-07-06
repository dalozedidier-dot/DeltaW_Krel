# DeltaW/K_rel reproducibility repository

This repository supports the DeltaW/K_rel manuscript with executable code,
tests, numerical validation artifacts, and submission-oriented documentation.
Its central purpose is to make the computational claims auditable: a reviewer
should be able to clone the repository, run the documented commands, and verify
which claims are supported by tests, SDP benchmarks, or exploratory simulations.

Public results page after GitHub Pages deployment:
`https://dalozedidier-dot.github.io/DeltaW_Krel/`.

## Scientific objective

The project studies a decision architecture around process-matrix witnesses,
relative robustness, and the relation between DeltaW-style witness signals and
K_rel-style causal robustness diagnostics. The repository separates three
levels of evidence:

1. **Confirmed SDP benchmarks**: exact process-matrix targets and the ideal
   quantum switch are checked with semidefinite programs.
2. **Methodological controls**: Monte Carlo and micro-tomography scripts test
   statistical and geometric behavior under simplified assumptions.
3. **Submission scaffolding**: manifests, preregistration locks, CI, notebooks,
   and claim matrices document what is reproducible and what remains editorial
   or archival.

The strongest external scientific check currently implemented is the ideal
quantum-switch generalized-robustness benchmark. The code reproduces the
published value near `0.5454` and uses the control-dephased switch as a negative
control with robustness near zero.

## Current scientific status

### Implemented and testable

- Linearized Monte Carlo control script for Annex C bis.
- Micro-tomography proof-of-concept / upper-bound stress test.
- Realistic finite-count tomography simulation bridge with multinomial/Poisson
  counts, regularized linear reconstruction, optional MLE, Fisher covariance,
  Ledoit-Wolf-style shrinkage, parametric bootstrap, and power maps.
- Full end-to-end tomography stress test with path-dependent losses, control
  dephasing, operation/control crosstalk, drift, interleaved controls,
  empirical LR calibration, and an applicability map.
- GitHub Pages dashboard publishing the SDP landscape and a finite-count full
  tomography smoke artifact for reviewer-facing inspection.
- Explicit trace-and-replace maps for the `[AI, AO, BI, BO]` convention.
- Projectors `L_V`, `L_AB`, `L_BA` with idempotence tests.
- Exact validation targets:
  - white-noise process;
  - fixed-order `A<=B` identity-channel process;
  - fixed-order `B<=A` identity-channel process;
  - causally separable convex mixtures.
- Minimal SDP over `K_CS = C_AB + C_BA` validating zero robustness on causally
  separable targets when `cvxpy` is available.
- **Ideal quantum switch implemented** in the Araujo-Branciard-Costa-Feix-
  Giarmatzi-Brukner convention (NJP 17, 102001 (2015)): rank-one process on
  `[AI, AO, BI, BO, F]` with `F = F_target tensor F_control` (D = 64), plus the
  with-future projectors `L_{A<=B<=F}`, `L_{B<=A<=F}` and `L_V^F`.
- **Published benchmark reproduced**: the generalized robustness of the ideal
  switch computed by `solve_switch_generalized_robustness` is approximately
  `0.545351` with SCS, matching the published `0.5454`; the control-dephased
  switch is causally separable within tolerance.
- **Partial-dephasing falsification scan available**: `scripts/run_switch_dephasing_scan.py`
  evaluates `W(lambda) = (1 - lambda) W_switch + lambda W_dephased` and exports
  the generalized-robustness curve with dual witness gaps and residuals.
- **Multi-family robustness landscape available**:
  `scripts/run_switch_robustness_landscape.py` scans control dephasing,
  white-noise visibility, and coherent order-bias families with the same SDP
  diagnostics.
- Full SDP diagnostics exported by `scripts/run_sdp_validation.py`: solver and
  package versions, iterations, solve time, residuals, minimal eigenvalues, and
  dual witness certificate.
- Unit/smoke tests and GitHub Actions CI with multi-Python coverage and a
  reproducibility pipeline.

### Interpretation guardrails

The Monte Carlo control bench and the micro-tomography script are a
**methodological/geometric control level and a simplified stress test**. They
are useful and reproducible, but they must not be presented as experimental
validation or as physical simulations of the switch. Only the SDP layer carries
the external published benchmark.

See [docs/SCIENTIFIC_CONTEXT.md](docs/SCIENTIFIC_CONTEXT.md) for a plain-language
explanation of the validation targets, robustness numbers, open parameter
directions, and what would count as going further scientifically.

## Installation status

This package has been installed and smoke-tested in the sandbox using:

```bash
python -m pip install -e .
pytest -q
python scripts/run_sdp_validation.py
```

See `docs/INSTALL_AND_TEST_REPORT.md` for the exact validation commands and
status. A local `.venv/` is intentionally **not** shipped in the ZIP, because
virtual environments should not be committed to GitHub.

## Repository layout

```text
.github/workflows/ci.yml              GitHub Actions tests
.github/workflows/pages.yml           GitHub Pages deployment
site/                                 static public results page
src/deltawkrel/projectors.py          trace-replace maps and process projectors
src/deltawkrel/sdp.py                 K_CS and switch robustness SDP routines
src/deltawkrel/switch_models.py       white-noise/fixed-order targets + switch
scripts/monte_carlo_control_supplement.py
scripts/micro_tomography_simulation.py
scripts/realistic_tomography_pipeline.py
scripts/full_tomography_simulation.py
scripts/full_realistic_tomography.py
scripts/run_sdp_validation.py
notebooks/projectors_definitions.ipynb
notebooks/validation_switch_ideal.ipynb
config/config_preregistration.json
tests/
docs/CLAIM_EVIDENCE_MATRIX.md
docs/MAXIMIZE_REPOSITORY_FOR_SUBMISSION.md
docs/ULTIMATE_VISION_ROADMAP.md
```

## Quick start

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
python -m pip install --upgrade pip
python -m pip install -e .
pytest -q
```

Run the proof-of-concept scripts. The smoke outputs are written under
`results/` so the generated reproducibility report can find them without path
translation:

```bash
python scripts/micro_tomography_simulation.py --outdir results/micro_smoke --n-sim 1000 --n-null 3000
python scripts/monte_carlo_control_supplement.py --n-sim 200 --n-null 500 --dim 4 --n-noise 1 --lambda-values 0,0.005 --n-values 1000 --output-dir results/mc_smoke --no-plots
```

Run the realistic finite-count tomography bridge:

```bash
python scripts/realistic_tomography_pipeline.py --outdir results/realistic_tomography_smoke --n-sim 50 --n-null 100 --n-boot 20
```

Run the full simulated tomography stress test:

```bash
python scripts/full_realistic_tomography.py --outdir results/full_tomography_smoke --n-sim 50 --n-null 100 --n-boot 20
```

Freeze the preregistration configuration and generate the reproducibility
report:

```bash
python scripts/freeze_preregistration.py
python scripts/generate_reproducibility_report.py
python scripts/validate_manifest.py
```

Run the SDP validation after installing `cvxpy`:

```bash
python scripts/run_sdp_validation.py
```

Run the partial control-dephasing scan for the switch:

```bash
python scripts/run_switch_dephasing_scan.py --lambdas 0,0.25,0.5,0.65,0.7,0.75,1
```

Run the broader robustness landscape:

```bash
python scripts/run_switch_robustness_landscape.py
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

The repository should be considered submission-ready only when all of the
following are true:

1. `projectors_definitions.ipynb` exports `L_V`, `L_AB`, `L_BA` superoperators
   and documents the exact equations/conventions.
2. `validation_switch_ideal.ipynb` constructs the ideal quantum switch and
   reproduces the published robustness benchmark.
3. Solver diagnostics are exported: status, objective value, residuals,
   `R_rel`, `omega_white`, eigenvalue checks, and witness-certificate checks.
4. The final repository is archived with a DOI, for example via Zenodo.
5. Every manuscript claim is mapped to executable evidence or explicitly marked
   as exploratory in `docs/CLAIM_EVIDENCE_MATRIX.md`.

For a claim-by-claim audit trail, see `docs/CLAIM_EVIDENCE_MATRIX.md`.
For the hardening roadmap, see `docs/MAXIMIZE_REPOSITORY_FOR_SUBMISSION.md`.
For the long-horizon research program, see
`docs/ULTIMATE_VISION_ROADMAP.md`.
