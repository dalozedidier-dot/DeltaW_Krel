# DeltaW/K_rel reproducibility repository

This repository is a GitHub-ready implementation scaffold for the ΔW/K_rel manuscript. It includes the toy/geometric Monte Carlo bench, a micro-tomography proof-of-concept, baseline process-matrix projectors, a minimal causal-SDP scaffold, tests, CI, and reproducibility templates.

## Current scientific status

What is ready:

- linearized Monte Carlo control script for Annex C bis;
- micro-tomography proof-of-concept / upper-bound stress test;
- baseline trace-and-replace maps and projector superoperators for `L_V`, `L_AB`, `L_BA`;
- executable SDP scaffold using `K_CS = C_AB + C_BA` on simple causally separable validation targets;
- unit/smoke tests and GitHub Actions CI.

What is **not** yet claim-complete:

- published-convention audit of `L_V`, `L_AB`, `L_BA` against the target equations;
- construction of the ideal quantum switch process matrix;
- reproduction of a published ideal-switch robustness benchmark.

Do not present the SDP validation as complete until `notebooks/validation_switch_ideal.ipynb` reproduces a known benchmark and the corresponding source/reference convention is fixed.

## Repository layout

```text
.github/workflows/ci.yml              GitHub Actions tests
src/deltawkrel/projectors.py          trace-replace maps and baseline projectors
src/deltawkrel/sdp.py                 minimal causal-SDP scaffold
src/deltawkrel/switch_models.py       safe placeholders for switch models
scripts/monte_carlo_control_supplement.py
scripts/micro_tomography_simulation.py
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

Run the proof-of-concept scripts:

```bash
python scripts/micro_tomography_simulation.py --outdir outputs --n-sim 1000 --n-null 3000
python scripts/monte_carlo_control_supplement.py --n-sim 200 --n-null 500 --dim 4 --n-noise 1 --lambda-values 0,0.005 --n-values 1000 --no-plots
```

Run the notebooks after installing Jupyter:

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

## Important methodological note

The scripts in this repository are useful for development and reproducibility, but the manuscript's confirmatory claim depends on the completed SDP/tomography validation. The micro-tomography and binomial simulations should be read as proof-of-concept / upper-bound stress tests, not full experimental predictions.
