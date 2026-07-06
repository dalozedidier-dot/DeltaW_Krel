# Certificate lemmas (task A2)

The two facts verified numerically to machine precision in
`docs/CERTIFIED_WITNESS_RESULT.md` are short theorems, not observations. Stated
at their natural generality they promote the result from "empirical scan" to
"certificate". Throughout, `H` is the space of Hermitian operators on a
finite-dimensional Hilbert space, with the real Hilbert-Schmidt inner product
`<A, B> = Tr(A B)` (operators are Hermitian, so this is real).

## Lemma 1 (affinity and single-scalar identifiability)

Let `W : [0,1] -> H` be an **affine family**, `W(l) = W0 + l*D` with
`D = W(1) - W(0)`. For any fixed `S in H` the witness signal

    g_S(l) := <S, W(l)> = <S, W0> + l * <S, D>

is affine in `l`. If `<S, D> != 0` the map `l -> g_S(l)` is a bijection onto its
range and

    l = ( g_S(l) - <S, W0> ) / <S, D> ,

so a **single real number** `g_S` determines `l`.

*Proof.* `<S, .>` is linear and `l -> W(l)` is affine, so their composition is
affine; a non-degenerate affine map of one real variable is invertible. ∎

*Remark.* The estimand is one scalar, independent of `dim H`. For the switch,
`dim H = 64` and the process has `dim^2 = 4096` real parameters, yet `l` is fixed
by one functional. This is the tomographic-collapse claim, and it is exact
(inversion error `~1e-16` in `scripts/run_certified_witness_analysis.py`).

## Lemma 2 (supporting hyperplane / certified lower bound)

Let the robustness be written in gauge (primal) form

    R(W) = min { Tr(X) / d_O : X in V,  W + X in C },

with `C` the closed convex cone of causally separable processes and `V` the
valid-process cone. Assume feasibility and a Slater point, so strong duality
holds. Then the Lagrange dual has the form

    R(W) = max { <S, W> : S in D },

where the **dual-feasible set `D` depends only on `C`, `V`, `d_O`** — not on `W`
(the process `W` enters only the linear dual objective). Let `S* in D` attain the
maximum at a reference `W_ref`, so `<S*, W_ref> = R(W_ref)`. Then

    <S*, W>  <=  R(W)   for every W in H,   with equality at W = W_ref.

Hence the affine functional `l_S*(W) = <S*, W>` is a **supporting hyperplane** of
the convex function `R` at `W_ref` (equivalently `S* in ∂R(W_ref)`).

*Proof.* `S*` is one feasible point of the dual maximisation, whose feasible set
does not depend on `W`; therefore for any `W`,
`<S*, W> <= max_{S in D} <S, W> = R(W)`. Equality at `W_ref` is strong duality.
`R` is a pointwise maximum of the linear functionals `<S, .>` over `S in D`, hence
convex; a linear functional lying below a convex function and touching it at a
point is a supporting hyperplane there. ∎

*Key point.* The lower bound `<S*, W> <= R(W)` needs **no affinity of the family**
— it holds for every `W`. Affinity (Lemma 1) is an extra property of *convex-
mixture* families. This is exactly the numerical pattern in
`run_certified_witness_landscape.py`: the lower bound holds for all three
families, while affinity holds only for `control_dephasing` and
`white_visibility` and fails (residual `~0.59`) for the pure-state `order_bias`
family.

## Corollary (certified nonseparability threshold)

Take an affine family with `W(0) = W_ref`. By Lemma 1,
`l_S*(l) = R(W_ref) + l * <S*, D>` is affine, and by Lemma 2 it lower-bounds
`R(W(l))` with equality at `l = 0`. If `R(W_ref) > 0` and `<S*, D> < 0`, its zero

    l* = - R(W_ref) / <S*, D>

satisfies `R(W(l)) >= l_S*(l) > 0` for all `l < l*`. Since `R(W) = 0` iff
`W in C` for these robustness measures, every `W(l)` with `l < l*` is **certified
causally nonseparable by the single fixed functional `S*`**, with no per-point
SDP and no full tomography. Furthermore `l* <= l_c`, where `l_c` is the true
boundary `R(W(l_c)) = 0`; the gap `l_c - l*` equals the curvature deficit of the
convex curve `R(W(l))` relative to its supporting line at `0`, and vanishes iff
`R` is affine along the family.

For `control_dephasing`: `R(W_ref) = R_g = 0.545351`, `<S*, D> = -0.771956`,
giving `l* = 0.7065` while `l_c = 1` — the gap `1 - 0.7065` is the price of a
single preregistered direction versus re-optimising the witness at each point.

## What this establishes

Lemma 1 + Lemma 2 + the corollary are the certificate: a **fixed, preregistered
affine functional** reproduces the robustness at the reference, lower-bounds it
everywhere (supporting hyperplane of a convex robustness), and yields a certified
threshold — all independent of Hilbert-space dimension. The numerics
(`affinity_residual ~2e-16`, `tightness_gap ~5e-11`, lower bound verified on the
SDP grid, `l* = 0.7065`) are confirmations of these statements, not the
statements themselves.
