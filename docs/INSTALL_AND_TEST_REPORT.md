# Installation and test report

This repository is installed and tested through the declared Python package
metadata and GitHub Actions workflows.

## Environment

- Supported Python versions in the extended CI workflow: 3.10, 3.11, 3.12.
- Install command:

```bash
python -m pip install -e .
```

The scientific stack includes numpy, scipy, pandas, matplotlib, cvxpy,
clarabel, pytest, pytest-cov, jupyterlab, nbformat, and tqdm.

## Validation commands

```bash
pytest -q tests
pytest tests --cov=src/deltawkrel --cov=scripts --cov-fail-under=90
python scripts/run_sdp_validation.py
python scripts/run_certified_witness_landscape.py
python scripts/run_certified_bounds.py
python scripts/run_finite_count_analysis.py
python scripts/micro_tomography_simulation.py --n-sim 50 --n-config 32 --outdir results/micro_smoke
python scripts/monte_carlo_control_supplement.py --n-sim 50 --n-null 200 --lambda-values 0,0.005 --n-values 1000 --no-plots --output-dir results/mc_smoke
python scripts/validate_manifest.py
```

## Results

- Unit and extended tests pass in the current CI artifacts.
- Coverage is above the required 90% gate in the extended workflow.
- Bipartite K_CS controls validate causally separable targets: white noise,
  fixed-order A before B, fixed-order B before A, and convex mixtures.
- The ideal quantum switch is implemented with an explicit future/control
  convention and reproduces the published generalized robustness benchmark near
  0.5454.
- The certified witness layer exports fixed-witness landscapes, an A5
  primal/dual interval, and an A4 finite-count single-scalar report.
- The control-dephased switch acts as a negative control with robustness near
  zero.
- Micro-tomography and Monte Carlo outputs are methodological stress tests, not
  experimental validation.

## Scientific status

The repository now provides a reproducible SDP benchmark for the ideal quantum
switch plus toy/geometric and micro-tomography stress tests. Before formal
submission, keep the manuscript, notebooks, generated CI artifacts, and archive
metadata synchronized with this implemented benchmark.

Do not include a local `.venv/` or installed site-packages in the GitHub
repository. The reproducible way to install is through `pyproject.toml`,
`requirements.txt`, or `environment.yml`.
