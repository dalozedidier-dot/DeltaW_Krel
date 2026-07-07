# Relation to prior work

This note records the positioning answer that a referee is likely to ask first:

```text
What does the DeltaW/K_rel program decide that the generalized-robustness SDP
and standard causal-nonseparability witnesses do not already decide?
```

The honest answer is deliberately narrow.

DeltaW/K_rel does **not** add detection power beyond the full causal SDP at the
reference point. In plain text: it does not add detection power beyond the full causal SDP.
The direction `S*` used by the certified layer is the
dual-optimal witness of the generalized-robustness SDP itself, and
`docs/CERTIFICATE_LEMMAS.md` states the supporting-hyperplane interpretation.
The contribution is therefore not "a stronger witness".

The repository supports three more specific contributions:

1. **A scalar decision fixed before data.** The functional `Tr(S* W)`, and after
   calibrated-noise projection `Tr(K_rel W)`, is selected before data analysis.
   This avoids post-hoc witness re-optimization.
2. **Calibrated-noise immunity by construction.** The admissible projection is
   orthogonal to preregistered nuisance directions. In the A4 finite-count
   artifact, the raw witness can be driven to near-certain false positives under
   calibrated drift while `K_rel` remains near the nominal alpha level.
3. **A reusable analytic certificate along affine families.** Once `S*` is
   solved at the reference switch, affine perturbation families inherit a
   certified lower-bound line without resolving the full SDP at every point.

Framed this way, the manuscript contribution is a preregistered,
calibrated-noise-robust decision protocol with a numerical certificate, not a
new frontier of detectable process matrices.

## Comparison

| Property | Full causal SDP / causal witness | Robustness measures | Causal inequalities | DeltaW/K_rel single-direction certificate |
|---|---|---|---|---|
| Certifies causal nonseparability | yes | yes | yes, when violated | yes |
| Detection power at reference | optimal | optimal | assumption-light but weaker for the switch | equal to `S*`, reduced after projection |
| Witness chosen before data | usually no | usually no | yes | yes |
| Requires full process tomography | usually yes | yes | no | estimates one fixed scalar; full tomography is not required for the scalar |
| Calibrated-noise immunity | not by construction | not by construction | depends on protocol | yes, for declared nuisance directions |
| Cost across a family | one SDP or witness optimization per point | one SDP per point | one inequality per correlation set | one reference solve plus certified scalar checks |
| Main limitation | witness can be retuned after data | same | lower power / different assumptions | no extra detection power; tightness can be lost by projection |

## Literature anchor

- Oreshkov, Costa, and Brukner, *Nature Communications* 3, 1092 (2012):
  process matrices and causal inequalities.
- Araujo, Branciard, Costa, Feix, Giarmatzi, and Brukner, *New Journal of
  Physics* 17, 102001 (2015): causal witnesses and the quantum-switch
  generalized-robustness benchmark reproduced in this repository.
- Branciard, *Scientific Reports* 6, 26018 (2016): causal witness case studies.
- Dourdent, Abbott, Brunner, and Branciard, *Physical Review Letters* 129,
  090402 (2022): semi-device-independent certification.
- Cao et al., *Optica* 10, 561 (2023): public semi-device-independent
  quantum-switch count file now verified in this repository as an external-data
  pilot.

## Abstract-safe wording

```text
We do not claim to certify processes beyond the optimal causal SDP. We convert
the SDP witness into a preregistered, single-valued, calibrated-noise-robust
decision rule with an explicit numerical certificate.
```
