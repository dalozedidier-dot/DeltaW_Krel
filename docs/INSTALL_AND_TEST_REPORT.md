# Installation and test report

This repository was installed and tested in the execution sandbox before packaging.

## Environment

- Python: 3.13.5 in the sandbox validation run
- Install command used:

```bash
python -m pip install -e .
```

The install pulled and/or verified the required scientific stack, including:

- numpy
- scipy
- pandas
- matplotlib
- cvxpy
- clarabel
- pytest
- jupyterlab / nbformat

## Validation commands executed

```bash
pytest -q
python scripts/run_sdp_validation.py
python scripts/micro_tomography_simulation.py --n-sim 50 --n-config 32 --outdir results/micro_smoke
python scripts/monte_carlo_control_supplement.py --n-sim 50 --n-null 200 --lambda-values 0,0.005 --n-values 1000 --no-plots --output-dir results/mc_smoke
```

## Results

- Unit and smoke tests: 12/12 passed.
- SDP infrastructure validation with CLARABEL: executed successfully.
- Micro-tomography smoke output: generated successfully.
- Monte Carlo control smoke output: generated successfully.

## Scientific status

The SDP infrastructure validates causally separable targets: white noise, fixed-order `A≺B`, fixed-order `B≺A`, and convex mixtures. The ideal quantum-switch benchmark remains intentionally marked as not implemented. The repository is therefore installed and technically testable, but not yet a complete submission-grade validation of the published quantum-switch robustness benchmark.

Do not include a local `.venv/` or installed site-packages in the GitHub repository. The reproducible way to install is through `pyproject.toml`, `requirements.txt`, or `environment.yml`.
