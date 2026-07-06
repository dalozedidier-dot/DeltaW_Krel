# Ultimate vision roadmap for DeltaW/K_rel

This document records the long-horizon ambition for the project without
promoting it to current evidence. The near-term repository remains anchored in
auditable SDP benchmarks, realistic simulated tomography, preregistration,
claim/evidence mapping, and reproducible artifacts.

This roadmap is an ambition tracker, not a current evidence claim.

## North star

Make DeltaW/K_rel a reference falsification protocol for proposed new physics
around quantum causality: indefinite causal order, relational constraints,
discrete quantum-gravity toy models, and post-quantum process theories.

The standard is deliberately high:

- preregistered decision rules before data;
- local negative controls and calibration locks;
- explicit causal witnesses and K_rel-style robustness diagnostics;
- finite-count tomography with covariance, bootstrap, and applicability gates;
- reproducible public code, manifests, release artifacts, and DOI archives;
- cautious claim language tied to the claim/evidence matrix.

## Level 1: go much further

### A. World-class numerical simulation

Target: a realistic end-to-end photonic quantum-switch simulator calibrated to
the level expected by experimental groups.

Required components:

- path-dependent losses and detector inefficiency;
- spatio-temporal dephasing, jitter, and distinguishability;
- source impurity and multi-photon contamination knobs;
- control-operation crosstalk and operation-dependent visibility loss;
- thermal/acoustic drift with interleaved calibration controls;
- higher-order process tomography with linear, constrained MLE, compressed
  sensing, Bayesian, and tensor-network options;
- massive Monte Carlo release jobs, from 10k smoke-scale maps to 100k-1M
  campaign-scale runs on cluster hardware.

Acceptance criterion:

- publish a sensitivity surface giving lambda_sens as a function of copies,
  visibility, crosstalk, drift, loss, reconstruction method, and LR threshold;
- report empirical false-positive rates under realistic miscalibration;
- keep a lightweight CI smoke test and store massive maps as release artifacts.

### B. Theoretical expansion

Target: make the protocol apply beyond a bipartite quantum switch.

Directions:

- multipartite switches and causal networks with three or more parties;
- causal witnesses for causal inequalities and quantum causal models;
- calibrated witnesses for resources beyond the current K_rel direction;
- semi-device-independent and, where possible, device-independent variants;
- toy links to relativistic or quantum-gravity-motivated causal-structure
  dynamics.

Acceptance criterion:

- each extension must ship with at least one analytic target, one negative
  control, one numerical benchmark, and one explicit applicability lock.

### C. Existing data and new experiments

Target: connect the protocol to real experimental practice.

Steps:

- inventory public indefinite-causal-order datasets and reproduce their stated
  processing assumptions where data are available;
- apply the DeltaW/K_rel pipeline as an exploratory a posteriori analysis, with
  all confirmatory language disabled;
- prepare one dedicated experimental design with shot budget, calibration
  cadence, estimated runtime, and failure modes;
- preregister a future confirmatory run through a public registry such as OSF;
- contact experimental teams only after the numerical readiness criteria are
  met.

Acceptance criterion:

- a public analysis notebook for each dataset family;
- one full preregistration package for a new run, including null controls and
  exclusion rules.

## Level 2: transformative program

| Axis | Ambition | Concrete product |
| --- | --- | --- |
| Theory | A theory of admissible relational signatures, not only one K_rel direction. | Formal paper plus benchmark examples. |
| Numerics | A community simulation, tomography, and causal-SDP platform. | Versioned Python package or future Julia package with DOI. |
| Experiment | A preregistered confirmatory test of a global relational constraint. | OSF preregistration, data release, and reproducibility bundle. |
| Interdisciplinary | Bridges to quantum gravity, relational quantum mechanics, causal sets, and operational reconstructions. | Perspective article and invited talks. |
| ML/AI | Learned witness discovery and protocol design under constraints. | Benchmark suite plus ablation studies against SDP witnesses. |
| Community | A shared falsification culture for causal/relational claims. | Workshop, tutorial notebooks, and public benchmark database. |

## Phased roadmap

### Phase 0: unbreakable foundation

Current repository target:

- full realistic tomography smoke artifact published on the site;
- all tests and coverage gates green;
- manifest validation green;
- claim/evidence matrix synchronized with the manuscript;
- arXiv/Zenodo-ready repository packaging.

### Phase 1: advanced proof of concept

Next research target:

- heavier tomography maps with 10k-50k null/bootstrap draws;
- interactive power and applicability maps on the site;
- compressed-sensing and Bayesian reconstruction comparisons;
- multi-witness cone-projection diagnostics;
- first public-data exploratory analyses.

### Phase 2: experimental design

Collaboration target:

- one optical or superconducting implementation proposal;
- realistic shot budget and runtime estimate;
- calibration schedule with interleaved controls;
- preregistration draft and data-management plan.

### Phase 3: community platform

Leadership target:

- package API stabilized around process models, tomography, SDP witnesses, and
  preregistered decision reports;
- benchmark database for indefinite causal order experiments;
- tutorial notebooks and a reproducible release workflow;
- workshop or tutorial session around falsifiable causal/relational tests.

## Non-negotiable guardrails

- Do not describe toy or smoke simulations as experimental evidence.
- Do not claim device independence without a matching protocol and assumptions.
- Do not infer new physics from a witness without local instrumental controls.
- Do not tune the admissible direction after looking at data.
- Do not publish massive maps without seeds, config, code version, and manifest.

## Immediate repository actions

1. Keep `scripts/full_realistic_tomography.py` as the public entry point for the
   end-to-end tomography stress test.
2. Expand the published Pages artifact from smoke-scale to release-scale maps.
3. Add interactive power/applicability visualizations once release-scale maps
   exist.
4. Add a `docs/EXPERIMENTAL_DESIGN_TEMPLATE.md` for future collaborators.
5. Add a public-data inventory document before any a posteriori analysis is
   framed as scientific evidence.
