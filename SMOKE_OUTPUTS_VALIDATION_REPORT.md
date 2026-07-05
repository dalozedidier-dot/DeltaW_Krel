# Smoke outputs validation report

## Status
Infrastructure smoke outputs are coherent for a development/test run. They are not scientific validation results.

## Files checked
- `outputs/micro_tomography_power.csv`: 12 rows
- `outputs/power_micro_tomography.png`: (1440, 900)
- `monte_carlo_outputs_control/basis_diagnostics.json`
- `monte_carlo_outputs_control/monte_carlo_power_results.csv/json`
- `monte_carlo_outputs_control/sdp_psd_proxy_diagnostics.json`
- `monte_carlo_outputs_control/simulation_config.json`
- `monte_carlo_outputs_control/witness_stability_diagnostics.json`

## Monte Carlo geometry diagnostics
- A_dim: 8.0
- projection_norm: 0.8628263508141821
- proj_norm_relative_to_random: 0.9151655454707268
- max_inner_N_K_rel: 3.469446951953614e-17

## Notes
- `sdp_check=false`: the SDP proxy was not run.
- `witness_stability_run=0`: the witness stability diagnostic was not run.
- JSON files have been sanitized to strict JSON: NaN values were replaced with null.

## Interpretation
These outputs validate that the smoke pipeline runs and exports coherent diagnostics. They do not validate the complete causal SDP, the ideal quantum switch benchmark, or full process-matrix tomography.
