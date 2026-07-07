# Public data inventory

This document expands the B1 inventory from
`docs/PROTOCOL_POSITIONING_AND_DATA_INVENTORY.md`. It separates three statuses:

- **verified public file**: the repository has parsed a public file and checked
  a reproducible numerical claim;
- **candidate for reanalysis**: the paper is relevant, but raw counts or a
  reconstructed process matrix have not been imported;
- **context only**: useful for positioning, but not an immediate data-ingestion
  target.

## Inventory

| Priority | Experiment | Platform | Reported quantity | Public-data status | DeltaW/K_rel path |
|---:|---|---|---|---|---|
| 1 | Cao et al., *Optica* 10, 561 (2023) | Photonic quantum switch, semi-device-independent certification | Semi-DI inequality value and experimental count table | **verified public file**: `scripts/ingest_external_cao2023_sdi.py` verifies `35,127,880` public counts from `jessicabavaresco/experimental-SDI-causality` | Correlation-level pilot only; not yet a process-matrix tomography reanalysis |
| 2 | Rubino et al., *Science Advances* 3, e1602589 (2017) | Photonic quantum switch | Causal witness of nonseparability | candidate for reanalysis; raw counts/process data not imported | Highest-priority process-matrix style target if tomographic data are available |
| 3 | Goswami et al., *Physical Review Letters* 121, 090503 (2018) | Photonic quantum switch | Causal witness / process characterization | candidate for reanalysis; raw data not imported | Strong candidate if reconstructed process matrix or counts can be obtained |
| 4 | Procopio et al., *Nature Communications* 6, 7913 (2015) | Photonic superposition of gate orders | Interferometric order-superposition signature | candidate/context; raw data not imported | Useful context, but not a direct full process-matrix dataset |
| 5 | Dourdent et al., *Physical Review Letters* 129, 090402 (2022) | Semi-device-independent theory/protocol | Semi-DI certification method | context only | Positions assumptions relative to DeltaW/K_rel |
| 6 | Additional quantum-switch task demonstrations | photonic/NMR/task-specific platforms | visibility, communication advantage, or task advantage | candidate/context; availability unresolved | Requires raw correlations or reconstructed process matrices before use |

## Verified public-data pilot: Cao 2023

The repository currently verifies the public file:

```text
https://github.com/jessicabavaresco/experimental-SDI-causality
```

Pinned inputs:

- source commit: `956a5ed3c8dcf00a7049acf3486ec8b391598863`
- raw file SHA-256:
  `dfe2739d794ccbb699861fb18640b815ebd175604f26d4781b1a21fb672d8a30`
- DOI: `10.1364/OPTICA.483876`
- arXiv: `2202.05346`

Verified outputs:

- `experimental_counts` shape: `[2, 2, 4, 2, 2, 2]`
- total counts: `35,127,880`
- recomputed `S_experiment`: `-0.0673482583497189`
- reported/recomputed agreement: below `1e-12`
- probability tables exactly match the normalized counts in the public file

Interpretation guardrail:

```text
This is real external experimental evidence for the Cao et al. semi-DI
inequality. It is not yet an E4 DeltaW/K_rel claim, because the file is a
correlation/count dataset rather than a process-matrix tomography dataset in the
repository convention.
```

## Minimum ingestion rule for future public files

Do not ingest screenshots or manually digitized plot points as evidence. A
candidate public dataset should enter the repository only when the following are
available or explicitly documented as unavailable:

1. DOI/arXiv, repository or archive URL, license, retrieval date, and immutable
   commit/hash where possible.
2. Raw file stored under `results/` for CI artifacts or a documented reason it
   cannot be redistributed.
3. Parser under `scripts/ingest_external_<source_id>.py`.
4. Unit tests using a local fixture, so CI does not depend only on network
   access.
5. A report under `site/data/external/` if the result is shown on GitHub Pages.
6. A convention audit before any DeltaW/K_rel process-matrix interpretation.
