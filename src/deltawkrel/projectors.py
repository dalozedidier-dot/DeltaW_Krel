r"""Projector utilities for bipartite process matrices.

Conventions
-----------
Subsystem order is [AI, AO, BI, BO].  The trace convention is
Tr(W)=d_O=d_AO*d_BO, as used in the manuscript.

The core operation is the standard process-matrix trace-and-replace map

    _S W := (I_S / d_S) \otimes Tr_S(W),

with tensor factors restored in the original order.

Implemented subspaces
---------------------
The bipartite valid-process projector is implemented as the composition of the
three standard linear constraints:

    [1-B_O] A_I A_O W = 0,
    [1-A_O] B_I B_O W = 0,
    [1-A_O][1-B_O] W = 0.

The fixed-order subspaces are implemented as the corresponding no-future-output
subspaces:

    L_AB(W): W = _B_O W with the A≺B process constraints,
    L_BA(W): W = _A_O W with the B≺A process constraints.

These maps are intended for the small-dimensional validation package.  They are
kept explicit and heavily tested so that convention changes can be audited.
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from itertools import product
from typing import Callable, Iterable, Sequence

import numpy as np

AI, AO, BI, BO = 0, 1, 2, 3
DEFAULT_LABELS = ("AI", "AO", "BI", "BO")


@dataclass(frozen=True)
class ProcessDims:
    """Dimensions for a bipartite process in order [AI, AO, BI, BO]."""

    AI: int = 2
    AO: int = 2
    BI: int = 2
    BO: int = 2

    @property
    def dims(self) -> tuple[int, int, int, int]:
        return (int(self.AI), int(self.AO), int(self.BI), int(self.BO))

    @property
    def hilbert_dim(self) -> int:
        out = 1
        for d in self.dims:
            out *= int(d)
        return int(out)

    @property
    def d_O(self) -> int:
        return int(self.AO * self.BO)


def _as_dims(dims: ProcessDims | Sequence[int]) -> tuple[int, int, int, int]:
    if isinstance(dims, ProcessDims):
        return dims.dims
    dims_tuple = tuple(int(d) for d in dims)
    if len(dims_tuple) != 4:
        raise ValueError("dims must have length 4 in order [AI, AO, BI, BO].")
    if any(d <= 0 for d in dims_tuple):
        raise ValueError("all subsystem dimensions must be positive.")
    return dims_tuple  # type: ignore[return-value]


def hilbert_dim(dims: ProcessDims | Sequence[int]) -> int:
    D = 1
    for d in _as_dims(dims):
        D *= int(d)
    return int(D)


def d_output(dims: ProcessDims | Sequence[int]) -> int:
    d = _as_dims(dims)
    return int(d[AO] * d[BO])


@lru_cache(maxsize=8192)
def _decode(index: int, dims: tuple[int, ...]) -> tuple[int, ...]:
    out = []
    x = int(index)
    for d in reversed(dims):
        out.append(x % d)
        x //= d
    return tuple(reversed(out))


@lru_cache(maxsize=8192)
def _encode(indices: tuple[int, ...], dims: tuple[int, ...]) -> int:
    x = 0
    for idx, d in zip(indices, dims):
        x = x * d + int(idx)
    return int(x)


def symmetrize(W: np.ndarray) -> np.ndarray:
    """Hermitian/symmetric part of W."""
    return 0.5 * (W + W.T.conj())


def trace_replace(W: np.ndarray, dims: ProcessDims | Sequence[int], systems: Iterable[int]) -> np.ndarray:
    """Apply _S W = (I_S/d_S) ⊗ Tr_S(W), preserving subsystem order.

    The implementation is explicit rather than clever. It is slow for large
    dimensions but very useful for auditability in d=2 validation tests.
    """
    dims_tuple = _as_dims(dims)
    systems_tuple = tuple(sorted(set(int(s) for s in systems)))
    if any(s < 0 or s >= len(dims_tuple) for s in systems_tuple):
        raise ValueError("systems contains an invalid subsystem index.")
    D = hilbert_dim(dims_tuple)
    W = np.asarray(W)
    if W.shape != (D, D):
        raise ValueError(f"W must have shape {(D, D)}, got {W.shape}.")
    if not systems_tuple:
        return W.copy()

    dS = int(np.prod([dims_tuple[s] for s in systems_tuple]))
    ranges = [range(dims_tuple[s]) for s in systems_tuple]
    out = np.zeros_like(W, dtype=np.result_type(W, float))

    for row in range(D):
        row_idx = _decode(row, dims_tuple)
        for col in range(D):
            col_idx = _decode(col, dims_tuple)
            # identity on replaced subsystems
            if any(row_idx[s] != col_idx[s] for s in systems_tuple):
                continue
            total = 0.0 + 0.0j if np.iscomplexobj(W) else 0.0
            for vals in product(*ranges):
                rr = list(row_idx)
                cc = list(col_idx)
                for s, val in zip(systems_tuple, vals):
                    rr[s] = int(val)
                    cc[s] = int(val)
                total += W[_encode(tuple(rr), dims_tuple), _encode(tuple(cc), dims_tuple)]
            out[row, col] = total / dS
    return out


def trrep(W: np.ndarray, dims: ProcessDims | Sequence[int], systems: Iterable[int]) -> np.ndarray:
    """Alias for the trace-and-replace map."""
    return trace_replace(W, dims, systems)


def _P_no_future_B(W: np.ndarray, dims: ProcessDims | Sequence[int]) -> np.ndarray:
    """Projector imposing [1-B_O] A_I A_O W = 0."""
    return W - trace_replace(W, dims, [AI, AO]) + trace_replace(W, dims, [AI, AO, BO])


def _P_no_future_A(W: np.ndarray, dims: ProcessDims | Sequence[int]) -> np.ndarray:
    """Projector imposing [1-A_O] B_I B_O W = 0."""
    return W - trace_replace(W, dims, [BI, BO]) + trace_replace(W, dims, [AO, BI, BO])


def _P_no_double_future(W: np.ndarray, dims: ProcessDims | Sequence[int]) -> np.ndarray:
    """Projector imposing [1-A_O][1-B_O] W = 0."""
    return (
        trace_replace(W, dims, [AO])
        + trace_replace(W, dims, [BO])
        - trace_replace(W, dims, [AO, BO])
    )


def L_valid_bipartite(W: np.ndarray, dims: ProcessDims | Sequence[int]) -> np.ndarray:
    """Orthogonal projector onto the bipartite valid-process linear subspace.

    This is implemented as the composition of the three commuting projectors
    associated with the standard linear constraints:

    * [1-B_O] A_I A_O W = 0,
    * [1-A_O] B_I B_O W = 0,
    * [1-A_O][1-B_O] W = 0.

    The trace/positivity constraints are checked separately.
    """
    X = _P_no_future_B(W, dims)
    X = _P_no_future_A(X, dims)
    X = _P_no_double_future(X, dims)
    return X


def L_A_before_B(W: np.ndarray, dims: ProcessDims | Sequence[int]) -> np.ndarray:
    """Projector for the fixed-order subspace A≺B.

    A before B is represented by processes satisfying W = _B_O W together with
    the no-signalling-to-A constraint [1-A_O] B_I B_O W = 0.  The composition
    below expands to the standard projective characterization

        L_AB(W) = _B_O W - _B_I B_O W + _A_O B_I B_O W,

    whose image lies inside the bipartite valid-process subspace.
    """
    X = _P_no_future_A(W, dims)
    return trace_replace(X, dims, [BO])


def L_B_before_A(W: np.ndarray, dims: ProcessDims | Sequence[int]) -> np.ndarray:
    """Projector for the fixed-order subspace B≺A.

    B before A is represented by processes satisfying W = _A_O W together with
    the no-signalling-to-B constraint [1-B_O] A_I A_O W = 0.  The composition
    below expands to

        L_BA(W) = _A_O W - _A_I A_O W + _A_I A_O B_O W.
    """
    X = _P_no_future_B(W, dims)
    return trace_replace(X, dims, [AO])


def white_noise_process(dims: ProcessDims | Sequence[int]) -> np.ndarray:
    """Normalized white-noise process with Tr(W)=d_O."""
    dims_tuple = _as_dims(dims)
    D = hilbert_dim(dims_tuple)
    return np.eye(D, dtype=float) * (d_output(dims_tuple) / D)


def maximally_entangled_projector(d_in: int, d_out: int | None = None) -> np.ndarray:
    """Unnormalized |Φ+><Φ+| on H_in⊗H_out.

    Requires equal dimensions unless d_out is explicitly equal.  The trace is d.
    """
    if d_out is None:
        d_out = d_in
    if int(d_in) != int(d_out):
        raise ValueError("identity-channel CJ projector requires matching dimensions.")
    d = int(d_in)
    phi = np.zeros((d * d,), dtype=float)
    for i in range(d):
        phi[i * d + i] = 1.0
    return np.outer(phi, phi)


def random_hermitian(dims: ProcessDims | Sequence[int], seed: int = 0) -> np.ndarray:
    D = hilbert_dim(dims)
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(D, D)) + 1j * rng.normal(size=(D, D))
    return symmetrize(X)


def is_psd(W: np.ndarray, *, atol: float = 1e-9) -> bool:
    vals = np.linalg.eigvalsh(symmetrize(np.asarray(W)))
    return bool(np.min(vals) >= -atol)


def is_idempotent(
    projector: Callable[[np.ndarray, ProcessDims | Sequence[int]], np.ndarray],
    dims: ProcessDims | Sequence[int],
    *,
    seed: int = 0,
    atol: float = 1e-8,
) -> bool:
    W = random_hermitian(dims, seed=seed)
    P = projector(W, dims)
    return bool(np.allclose(projector(P, dims), P, atol=atol))


def assert_trace_convention(W: np.ndarray, dims: ProcessDims | Sequence[int], *, atol: float = 1e-8) -> None:
    target = d_output(dims)
    if not np.allclose(np.trace(W), target, atol=atol):
        raise AssertionError(f"Trace convention violated: Tr(W)={np.trace(W)} != d_O={target}.")


def superoperator_matrix(
    projector: Callable[[np.ndarray, ProcessDims | Sequence[int]], np.ndarray],
    dims: ProcessDims | Sequence[int],
    *,
    order: str = "C",
) -> np.ndarray:
    """Return P such that vec(projector(W)) = P vec(W)."""
    D = hilbert_dim(dims)
    N = D * D
    P = np.zeros((N, N), dtype=float)
    for k in range(N):
        E = np.zeros((D, D), dtype=float)
        if order == "C":
            i, j = divmod(k, D)
        elif order == "F":
            j, i = divmod(k, D)
        else:
            raise ValueError("order must be 'C' or 'F'.")
        E[i, j] = 1.0
        P[:, k] = np.asarray(projector(E, dims), dtype=float).reshape(-1, order=order)
    return P
