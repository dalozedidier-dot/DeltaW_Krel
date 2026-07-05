# SDP validation status

## Implemented now

- Explicit trace-and-replace maps for the bipartite convention `[AI, AO, BI, BO]`.
- Projectors `L_V`, `L_AB`, `L_BA` based on the standard bipartite linear constraints.
- Idempotence tests for `L_V`, `L_AB`, `L_BA` and their vectorized superoperators.
- Exact causally separable validation targets:
  - white-noise process;
  - fixed-order `A≺B` identity-channel process;
  - fixed-order `B≺A` identity-channel process;
  - convex mixture of both fixed orders.
- Minimal SDP over `K_CS = C_AB + C_BA` validating that those causally separable targets have zero robustness, when `cvxpy/CLARABEL` is available.

## Ideal quantum-switch benchmark

The ideal quantum switch is implemented in `src/deltawkrel/switch_models.py`
using the Araújo, Branciard, Costa, Feix, Giarmatzi, Brukner convention
(New J. Phys. 17, 102001 (2015)): systems `[AI, AO, BI, BO, F]`, with
`F = F_target ⊗ F_control`, target state `|0>`, and control state `|+>`.

`scripts/run_sdp_validation.py` exports:

1. the bipartite causally separable controls listed above;
2. generalized robustness for the ideal switch, benchmarked against the
   published reference value near `0.5454`;
3. random-robustness diagnostics;
4. a control-dephased switch negative control with robustness near zero;
5. solver/version, residual, eigenvalue, and witness-certificate diagnostics.

The remaining submission work is editorial and archival: keep the notebook,
manuscript wording, generated CI artifacts, and DOI metadata synchronized with
this implemented benchmark.
