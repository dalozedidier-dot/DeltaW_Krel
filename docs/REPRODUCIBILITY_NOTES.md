# Reproducibility notes

This repository distinguishes three levels:

1. **Toy/geometric bench** — implemented in `scripts/monte_carlo_control_supplement.py`.
2. **Micro-tomography proof-of-concept** — implemented in `scripts/micro_tomography_simulation.py`.
3. **Full causal-SDP/tomography validation** — scaffolded in `src/deltawkrel/` and notebooks, but the ideal-switch benchmark remains to be completed before formal submission.

The baseline process-matrix projectors are executable and tested for idempotence. They must still be audited against the exact target convention before being used for confirmatory claims.


## Canonical smoke-output layout

The reproducibility pipeline writes regenerated smoke artifacts under `results/`:

- `results/micro_smoke/micro_tomography_power.csv`;
- `results/micro_smoke/power_micro_tomography.png`;
- `results/mc_smoke/monte_carlo_power_results.json`;
- `results/mc_smoke/monte_carlo_power_results.csv`;
- `results/mc_smoke/basis_diagnostics.json`;
- `results/mc_smoke/witness_stability_diagnostics.json` when the stability diagnostic is enabled.

Older local bundles may still contain `outputs/` and `monte_carlo_outputs_control/`.
`generate_reproducibility_report.py` reports those as `legacy-present`, but new CI and Makefile targets use the canonical `results/` paths.
