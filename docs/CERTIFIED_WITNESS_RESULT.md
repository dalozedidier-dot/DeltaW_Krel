# Certified single-direction causal witness (E3+)

This note records the strongest single-author, no-laboratory result the
DeltaW/K_rel program currently supports. It fuses the SDP benchmark layer and
the preregistered decision layer on the **physical** switch process, and turns
the reproduction of a known number into a certified falsification statement.

Artifacts: `src/deltawkrel/certified_witness.py`,
`scripts/run_certified_witness_analysis.py`,
`tests/test_certified_witness.py`,
`results/certified_witness_report.json`, `.csv`, `.png`.

## Statement

Let `W_switch` be the ideal quantum switch and `S*` the dual-optimal causal
witness of its generalized-robustness SDP. For the physical dephasing family
`W(lambda) = (1 - lambda) W_switch + lambda W_dephased`:

1. **Benchmark.** `R_g(W_switch) = 0.545351`, reproducing the published `0.5454`
   (Araujo et al., NJP 17, 102001 (2015)); strong duality holds,
   `|Tr(S* W_switch) - R_g| ~ 5e-11`.
2. **Affinity.** `lambda -> Tr(S* W(lambda))` is affine to machine precision
   (max residual `~2e-16`). This is not a fit: `W(lambda)` is affine in
   `lambda` and the witness functional is linear.
3. **Certified lower bound.** Because the dual-feasible set of the SDP does not
   depend on `W`, `Tr(S* .)` is the supporting affine functional at
   `W_switch`: it equals `R_g` at `lambda = 0` and lower-bounds the full SDP
   `R_g(lambda)` at every point (verified against the SDP on a grid; the SDP
   curve is convex, as it must be).
4. **Certified threshold.** The zero crossing `lambda* = 0.7065` is a certified
   causal-nonseparability threshold: for `lambda < lambda*`,
   `R_g(lambda) >= Tr(S* W(lambda)) > 0`, so `W(lambda)` is provably causally
   nonseparable using **one fixed scalar**, with no fresh SDP and no full
   tomography at each point. `lambda* <= lambda_c` (the true separability point,
   here `lambda_c = 1`); the gap `1 - lambda*` is the price of committing to a
   single preregistered direction instead of re-optimising the witness.
5. **Tomographic collapse.** A single expectation `<S*>` inverts `lambda`
   exactly in the noiseless case (recovery error `~2e-16`), replacing the
   `D^2 = 4096` real parameters of the `D = 64` process matrix with one number.
6. **Relational direction.** `K_rel = P_A(S*)`, the projection of `S*` onto the
   complement of the preregistered calibrated-noise directions (here identity /
   normalization and target depolarization), is first-order blind to those
   directions (leakage `~1e-17`) while still responding to the falsification
   axis. It keeps `56.8 %` of the certificate norm and sits at
   `cos(angle) = 0.568` from `S*`: this angle is the quantified cost of
   relational robustness.

## What this is and is not

This is **E3+**: a certified mathematical statement on the physical switch,
above the bare benchmark reproduction because it (i) is a lower-bound
certificate rather than a single number, (ii) instantiates the preregistered
K_rel rule on the real process instead of a toy space, and (iii) quantifies the
tomographic and noise-robustness tradeoffs exactly.

It is **not** E4. The noiseless single-scalar inversion and the noise immunity
are analytic/simulated properties. Turning them into an experimental claim
requires finite-count statistics with a calibrated measurement model and a
preregistered refutation rule on real or archival data, per
`docs/CLAIM_EVIDENCE_MATRIX.md` and the acceptance criteria in
`docs/ULTIMATE_VISION_ROADMAP.md`.

## Multi-family landscape

The **same** fixed witness `S*`, extracted once at the ideal switch, certifies
all three E3 perturbation families. This is the strength of preregistration: one
direction, chosen before data, works across the landscape.

| Family | Type | Certified region | Single scalar recovers |
| --- | --- | --- | --- |
| `control_dephasing` `W(lambda)` | affine (ref `lambda=0`) | `lambda in (0, 0.706)` | `lambda` exactly (one-sided) |
| `white_visibility` `W(v)` | affine (ref `v=1`) | `v in (0.507, 1)` | `v` exactly (one-sided) |
| `order_bias` `W(q)` | nonlinear (ref `q=1/2`) | `q in (0.024, 0.976)` | order-coherence magnitude only |

Affinity residual is near machine precision for the two convex-mixture families
and large for the pure-state order-bias family; the map is genuinely nonlinear
there. The lower bound `Tr(S* W) <= R_g(W)` holds on every verification grid for
all three families, because `S*` is dual-feasible regardless of `W`.

The honest structural finding: for the amplitude-biased order family the
witness is symmetric about `q=1/2`, so one scalar `<S*>` measures how balanced
the order superposition is but not which order dominates. Resolving direction
requires a second, order-asymmetric witness.

Artifacts: `scripts/run_certified_witness_landscape.py`,
`site/data/certified_witness/certified_witness_landscape.json`, `.csv`, `.png`.

## Reproduce

```bash
python scripts/run_certified_witness_analysis.py
python scripts/run_certified_witness_landscape.py
pytest tests/test_certified_witness.py -q
```
