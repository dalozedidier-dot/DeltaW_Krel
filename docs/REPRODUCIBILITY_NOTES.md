# Reproducibility notes

This repository distinguishes three levels:

1. **Toy/geometric bench** — implemented in `scripts/monte_carlo_control_supplement.py`.
2. **Micro-tomography proof-of-concept** — implemented in `scripts/micro_tomography_simulation.py`.
3. **Full causal-SDP/tomography validation** — scaffolded in `src/deltawkrel/` and notebooks, but the ideal-switch benchmark remains to be completed before formal submission.

The baseline process-matrix projectors are executable and tested for idempotence. They must still be audited against the exact target convention before being used for confirmatory claims.
