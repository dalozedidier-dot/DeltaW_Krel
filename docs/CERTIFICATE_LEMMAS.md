# Certificate Lemmas

These two facts are the mathematical core behind the certified witness layer in
`docs/CERTIFIED_WITNESS_RESULT.md`. They turn the numerical SDP output into a
certificate: one fixed functional is extracted at the ideal switch, then reused
as a preregistered lower bound on nearby process families.

Throughout, `H` is the real vector space of Hermitian operators on a
finite-dimensional Hilbert space, with Hilbert-Schmidt inner product
`<A, B> = Tr(A B)`.

## Lemma 1: Affinity and Single-Scalar Identifiability

Let `W : [0,1] -> H` be an affine family,

```text
W(lambda) = W0 + lambda * D,    D = W(1) - W(0).
```

For any fixed `S in H`, the witness signal

```text
g_S(lambda) = <S, W(lambda)> = <S, W0> + lambda * <S, D>
```

is affine in `lambda`. If `<S, D> != 0`, the map `lambda -> g_S(lambda)` is
invertible on its range:

```text
lambda = (g_S(lambda) - <S, W0>) / <S, D>.
```

So one real number `g_S` identifies the one-dimensional mixture parameter.

Proof: `<S, .>` is linear and `lambda -> W(lambda)` is affine, so their
composition is affine. A non-degenerate affine map of one real variable is
invertible. QED.

For the switch implementation, the process matrix has `D = 64` and therefore
`D^2 = 4096` real matrix entries, but the preregistered dephasing parameter is
identified by one functional along an affine family. The finite-count module
`src/deltawkrel/finite_count.py` quantifies the resulting one-parameter
shot-noise scaling.

## Lemma 2: Supporting Hyperplane and Certified Lower Bound

Write generalized robustness in gauge form:

```text
R(W) = min { Tr(X) / d_O : X in V, W + X in C },
```

where `C` is the closed convex cone of causally separable processes and `V` is
the valid-process cone. Under feasibility and a Slater point, strong duality
gives the dual form

```text
R(W) = max { <S, W> : S in D },
```

where the dual-feasible set `D` depends on the cones and normalization, not on
the specific process `W`. If `S* in D` attains the maximum at a reference
process `W_ref`, then

```text
<S*, W> <= R(W)          for every W,
<S*, W_ref> = R(W_ref).
```

Thus `W -> <S*, W>` is a supporting affine functional of the convex robustness
function at `W_ref`.

Proof: `S*` is dual-feasible for every process because the dual feasible set is
process-independent. Therefore `<S*, W>` is bounded above by the dual optimum,
which equals `R(W)` by strong duality. Equality at `W_ref` is the definition of
optimality at the reference. QED.

This lower-bound property does not require the process family to be affine. The
affinity claim belongs to Lemma 1; the conic lower bound belongs to Lemma 2.
This distinction is visible in `scripts/run_certified_witness_landscape.py`:
control dephasing and white visibility are affine single-scalar families, while
coherent order bias is nonlinear and two-sided, yet the fixed witness remains a
valid certified lower bound on the verification grid.

## Corollary: Certified Nonseparability Threshold

For an affine family with `W(0) = W_ref`, Lemma 1 gives

```text
<S*, W(lambda)> = R(W_ref) + lambda * <S*, D>.
```

By Lemma 2 this affine quantity lower-bounds `R(W(lambda))` everywhere. If
`R(W_ref) > 0` and `<S*, D> < 0`, then

```text
lambda_star = -R(W_ref) / <S*, D>
```

is a certified threshold: every `lambda < lambda_star` has
`R(W(lambda)) > 0`, hence is causally nonseparable for this robustness witness.

For the implemented control-dephasing family:

```text
R(W_ref) = 0.545351
<S*, D>  = -0.771956
lambda_star ~= 0.7065
```

The true reoptimized SDP boundary for total dephasing is at `lambda = 1`. The
gap is the price of using one fixed preregistered functional instead of
retuning the witness at each point.

## Executable Checks

- A2 proofs: this document.
- A4 finite-count statistics: `src/deltawkrel/finite_count.py`,
  `scripts/run_finite_count_analysis.py`, and `tests/test_finite_count.py`.
- A5 certified interval: `src/deltawkrel/certified_bounds.py` and
  `scripts/run_certified_bounds.py`.

The current A5 report brackets the ideal-switch robustness as
`R_g in [0.545351058860, 0.545351059392]` using the SCS primal/dual values, with
CLARABEL and MOSEK status reported explicitly.
