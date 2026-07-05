# Scientific context and interpretation guide

This note explains what the current repository proves, what it only tests as a
software or methodological control, and what would count as a deeper scientific
extension.

## What is validated now

The strongest validation layer is the SDP layer. It checks process matrices
against causal-order cones and reproduces a known benchmark for the ideal
quantum switch.

| Target | Role | Expected result |
| --- | --- | --- |
| White-noise process | Neutral baseline inside the valid process cone. | Zero causal robustness. |
| Fixed-order `A<=B` identity process | Known causally ordered control. | Zero causal robustness. |
| Fixed-order `B<=A` identity process | Same control with the opposite order. | Zero causal robustness. |
| Convex mixture of fixed orders | Known causally separable process. | Zero causal robustness. |
| Ideal quantum switch | Published nonseparable benchmark. | Generalized robustness near `0.5454`. |
| Control-dephased switch | Negative control removing coherent order superposition. | Robustness near zero. |

When the SDP status is `optimal`, it means the numerical solver found a
solution satisfying the cone and equality constraints within its declared
tolerances. It is not just a software success flag: it supports the mathematical
claim only when the convention, dimensions, residuals, eigenvalues, and witness
diagnostics are also reported.

## What the smoke tests do and do not prove

The Monte Carlo and micro-tomography scripts are intentionally lightweight.
They are useful because they make the pipeline executable, catch regressions,
exercise JSON/report generation, and illustrate how statistical power or witness
behavior could be studied.

They do **not** prove final experimental feasibility. They should be described
as methodological controls or proof-of-concept stress tests unless they are
extended with a physical noise model, calibrated measurement model, and a
predeclared inference target.

## Open scientific directions

The current configuration `quantum_switch_d2` is enough to validate the
published ideal-switch benchmark. Going further means exploring controlled
departures from this reference case:

1. **Noise families**: white noise, control dephasing, target depolarization,
   coherent phase errors, and biased order probabilities.
2. **Dimension and system size**: compare the current qubit-level benchmark
   with higher-dimensional or reduced effective models when the conventions are
   mathematically defined.
3. **Tomography burden**: document how sample size, measurement design, and
   reconstruction assumptions affect witness reliability.
4. **Solver cross-checks**: compare SCS with another solver such as MOSEK when
   available, and preserve tolerances, residuals, and version metadata.
5. **Witness interpretation**: connect each reported robustness value to the
   manuscript claim it supports, using `docs/CLAIM_EVIDENCE_MATRIX.md`.

## Practical recommendation for publication

For a public repository or soutenance package, present the work in three layers:

1. **Benchmark layer**: SDP reproduction of the ideal quantum-switch robustness.
2. **Control layer**: causally separable targets and dephased-switch negative
   control.
3. **Exploratory layer**: Monte Carlo and micro-tomography smoke tests, clearly
   labelled as methodological rather than final physical evidence.

This framing keeps the central scientific claim strong while avoiding
overclaiming from the toy simulations.
