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

## Still a submission blocker

The ideal quantum switch process is **not** implemented. The function `ideal_quantum_switch_process()` raises `NotImplementedError` on purpose. This prevents the repository from accidentally claiming a completed benchmark.

Before formal submission, the repository must add:

1. explicit ideal-switch process matrix in the selected convention, including the future/control system if used;
2. comparison against a published benchmark;
3. exported solver diagnostics for that benchmark;
4. documented convention mapping between the manuscript equations and the implementation.

Until then, the repository is suitable for implementation/testing, but not yet a final submission supplement.
