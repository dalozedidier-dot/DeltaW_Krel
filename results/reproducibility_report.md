# DeltaW/K_rel reproducibility report

This report is generated from repository artifacts. It distinguishes
implemented evidence from explicit submission blockers.

## Artifact status

- Preregistration config: present (`config/config_preregistration.json`)
- Preregistration lock: present (`results/preregistration_lock.json`)
- SDP validation report: present (`results/sdp_validation_report.json`)
- Monte Carlo smoke output: present (`results/mc_smoke/monte_carlo_power_results.json`)
- Micro-tomography smoke output: present (`results/micro_smoke/micro_tomography_power.csv`)
- Claim/evidence matrix: present (`docs/CLAIM_EVIDENCE_MATRIX.md`)

## Preregistration lock

- status: locked
- scenario: quantum_switch_bipartite_d2
- trace convention: Tr(W)=d_O
- config sha256: `21a6e4cce29fcc3069c43bc597a8101053fa6cba46aa6893dd4485af515d0a77`

## SDP validation

- status: infrastructure_validation_only
- ideal quantum switch implemented: False
- ideal quantum switch submission blocker: True
- white_noise: optimal (objective=0.0, solver=CLARABEL)
- fixed_order_A_before_B: optimal (objective=2.1788500930243524e-10, solver=CLARABEL)
- fixed_order_B_before_A: optimal (objective=2.1788366435426067e-10, solver=CLARABEL)
- causally_separable_mixture_q037: optimal (objective=0.0, solver=CLARABEL)

## Manifest

- manifest entries: 42
- validation command: `python scripts/validate_manifest.py`

## Submission status

The repository currently supports toy/geometric simulations,
micro-tomography proof of concept, process-matrix projector tests,
and K_CS infrastructure validation. The ideal quantum-switch
benchmark remains the decisive submission blocker until implemented
and compared with a published reference value.
