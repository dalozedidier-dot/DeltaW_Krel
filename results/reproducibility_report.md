# DeltaW/K_rel reproducibility report

This report is generated from repository artifacts. It distinguishes
implemented evidence from explicit submission blockers.

## Artifact status

- Preregistration config: present (`config/config_preregistration.json`)
- Preregistration lock: present (`results/preregistration_lock.json`)
- SDP validation report: present (`results/sdp_validation_report.json`)
- Monte Carlo smoke output: missing (`results/mc_smoke/monte_carlo_power_results.json`)
- Micro-tomography smoke output: missing (`results/micro_smoke/micro_tomography_power.csv`)
- Claim/evidence matrix: present (`docs/CLAIM_EVIDENCE_MATRIX.md`)

## Preregistration lock

- status: locked
- scenario: quantum_switch_bipartite_d2
- trace convention: Tr(W)=d_O
- config sha256: `21a6e4cce29fcc3069c43bc597a8101053fa6cba46aa6893dd4485af515d0a77`

## SDP validation

- status: sdp_validation_with_switch_benchmark
- ideal quantum switch implemented: True
- ideal quantum switch submission blocker: False
- switch generalized robustness: 0.5453510590583036 (reference 0.5454, passed=True)
- white_noise: optimal (objective=0.0, solver=CLARABEL)
- fixed_order_A_before_B: optimal (objective=2.1788500930243524e-10, solver=CLARABEL)
- fixed_order_B_before_A: optimal (objective=2.1788366435426067e-10, solver=CLARABEL)
- causally_separable_mixture_q037: optimal (objective=0.0, solver=CLARABEL)

## Manifest

- manifest entries: 44
- validation command: `python scripts/validate_manifest.py`

## Submission status

The repository supports toy/geometric simulations (methodological
control bench, NOT experimental validation), the micro-tomography
proof of concept, process-matrix projector tests, and K_CS SDP
validation. The ideal quantum switch is implemented in the
Araújo et al. (NJP 17, 102001 (2015)) convention; its generalized
robustness benchmark (reference 0.5454) is REPRODUCED. See docs/CONVENTIONS.md for the official convention and the
equation-to-code audit table.
