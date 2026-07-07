# Protocol Positioning and External Data Inventory

This document covers two Phase A/B deliverables:

- A3: how DeltaW/K_rel should be positioned relative to existing indefinite
  causal order tests.
- B1: which published quantum-switch datasets or papers should be targeted for
  future external-data reanalysis.

One external public experimental count file has now been verified
reproducibly: Cao et al. 2023 provides a semi-device-independent
quantum-switch MATLAB file with `experimental_counts`, `p_experiment`,
`p_theory`, and inequality coefficients. This is not a DeltaW/K_rel
process-matrix tomography reanalysis yet. The remaining entries below are a
prioritized inventory for future reproducible ingestion.

## A3: Positioning

DeltaW/K_rel is not another post-hoc causal witness fit. Its current strongest
claim is narrower and cleaner:

1. Extract one dual-optimal witness `S*` at the ideal quantum switch.
2. Certify, by conic duality, that `Tr(S* W)` is a lower bound on generalized
   robustness for every process.
3. Project the witness onto a preregistered calibrated-noise complement to form
   `K_rel`.
4. Use finite-count statistics only for the resulting scalar, not for an
   unconstrained `D^2` process reconstruction claim.

This makes the protocol complementary to existing causal-witness and
semi-device-independent certifications:

- It is more conservative than re-optimizing a witness after data are seen.
- It is less assumption-light than semi-device-independent or
  device-independent protocols.
- It is stronger than toy Monte Carlo because the central benchmark is the full
  switch SDP and a primal/dual certificate interval.
- It remains experiment-facing because A4 quantifies finite-count behavior and
  the realistic tomography scripts quantify losses, crosstalk, drift, and
  reconstruction effects.

Recommended manuscript wording:

```text
We do not claim device independence. We provide a preregistered,
solver-certified, single-functional falsification protocol for process-matrix
data, designed to be robust against specified calibrated nuisance directions.
```

## Current Internal Evidence Stack

| Layer | Artifact | Status | Claim strength |
|---|---|---:|---|
| SDP benchmark | `src/deltawkrel/sdp.py` | implemented/tested | E3 |
| Fixed dual witness | `src/deltawkrel/certified_witness.py` | implemented/tested | E3+ |
| Certified interval | `src/deltawkrel/certified_bounds.py` | implemented/tested | E3+ |
| Finite-count scalar estimator | `src/deltawkrel/finite_count.py` | implemented/tested | E2+ |
| Realistic tomography stress | `scripts/full_realistic_tomography.py` | smoke artifact | E2 |
| External public counts | `scripts/ingest_external_cao2023_sdi.py` | Cao 2023 semi-DI counts verified | external-data pilot |
| External DeltaW/K_rel reanalysis | future ingestion | not implemented | not claimed |

## B1: External Quantum-Switch Inventory

| Priority | Reference | DOI / arXiv | Relevance | Current data status |
|---:|---|---|---|---|
| 1 | Procopio et al., "Experimental Superposition of Orders of Quantum Gates", Nat. Commun. 6:7913 (2015) | DOI `10.1038/ncomms8913`, arXiv `1412.4006` | early photonic superposition-of-orders experiment; useful for gate-order task context | paper located; raw data not imported |
| 2 | Rubino et al., "Experimental Verification of an Indefinite Causal Order", Sci. Adv. 3:e1602589 (2017) | DOI `10.1126/sciadv.1602589`, arXiv `1608.01683` | causal-witness experiment; high priority for exploratory DeltaW/K_rel reanalysis | paper located; raw counts/process data not imported |
| 3 | Goswami et al., "Indefinite Causal Order in a Quantum Switch", Phys. Rev. Lett. 121, 090503 (2018) | DOI `10.1103/PhysRevLett.121.090503`, arXiv `1803.04302` | photonic quantum switch with causal witness; directly aligned with the implemented switch benchmark | paper located; raw data not imported |
| 4 | Dourdent et al., "Semi-Device-Independent Certification of Causal Nonseparability with Trusted Quantum Inputs", Phys. Rev. Lett. 129, 090402 (2022) | DOI `10.1103/PhysRevLett.129.090402`, arXiv `2107.10877` | theoretical/semi-DI bridge; positions DeltaW/K_rel against lower-assumption certifications | theory target; no raw experimental data expected |
| 5 | Cao et al., "Semi-device-independent certification of indefinite causal order in a photonic quantum switch", Optica 10, 561 (2023) | DOI `10.1364/OPTICA.483876`, arXiv `2202.05346`; code/data repository `https://github.com/jessicabavaresco/experimental-SDI-causality` | experimental semi-DI switch; strong target for external-data comparison | public MATLAB count file verified by `scripts/ingest_external_cao2023_sdi.py`; not yet converted to DeltaW/K_rel |
| 6 | Richter, Antesberger, Cao, Walther, Rozema, "Towards an Experimental Device-Independent Verification of Indefinite Causal Order" (2025 preprint) | arXiv `2506.16949` | closest verified Antesberger-linked experimental item found in search; not the same as "Antesberger 2024" | preprint located; dataset status unresolved |
| 7 | Valibouse et al., "Time-Delocalized Local Measurements in an Indefinite Causal Order" (2026 preprint) | arXiv `2604.11878` | new time-delocalized local measurement scheme; possible future stress case | preprint located; dataset status unresolved |

## Ingestion Plan

For any external dataset, the repository should not ingest screenshots or
manually transcribed plot points as evidence. The minimum acceptable path is:

1. Record source DOI/arXiv, data license, file URL, and retrieval date.
2. Store immutable raw files under `external_data/raw/<source_id>/` or document
   why they cannot be redistributed.
3. Add a parser under `scripts/ingest_external_<source_id>.py`.
4. Convert counts/correlations into the repository's process/witness convention
   with an explicit convention audit.
5. Run the preregistered `S*` and `K_rel` diagnostics without post-hoc retuning.
6. Mark the result exploratory unless the dataset was selected and locked before
   analysis.

## Guardrail

Until raw experimental data are ingested and parsed through the DeltaW/K_rel
convention reproducibly, the site and manuscript should say:

```text
One external semi-DI public count file has been verified as a pilot. Full
DeltaW/K_rel external-data reanalysis is planned, but not yet claimed.
```
