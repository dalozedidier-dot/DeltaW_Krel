"""Projector utilities for bipartite process matrices.

Conventions
-----------
Subsystem order is [AI, AO, BI, BO].  The trace convention used throughout the
manuscript is Tr(W)=d_O=d_AO*d_BO.

The maps implemented here use the standard process-matrix replacement notation
_S W := (I_S/d_S) ⊗ Tr_S(W), with tensor factors restored in the original order.

Important status
----------------
L_valid_bipartite, L_A_before_B, and L_B_before_A are implemented as a
reproducible baseline for validation tests.  Before formal submission, the exact
convention must be cross-checked against the equations used in the target paper
(Oreshkov-Costa-Brukner 2012 and follow-up witness papers), and the notebook
`validation_switch_ideal.ipynb` must reproduce a published benchmark.
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
        return (self.AI, self.AO, self.BI, self.BO)

    @property
    def hilbert_dim(self) -> int:
        out = 1
        for d in self.dims:
            out *= int(d)
        return out

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


@lru_cache(maxsize=4096)
def _decode(index: int, dims: tuple[int, ...]) -> tuple[int, ...]:
    out = []
    x = int(index)
    for d in reversed(dims):
        out.append(x % d)
        x //= d
    return tuple(reversed(out))


@lru_cache(maxsize=4096)
def _encode(indices: tuple[int, ...], dims: tuple[int, ...]) -> int:
    x = 0
    for idx, d in zip(indices, dims):
        x = x * d + int(idx)
    return int(x)


def symmetrize(W: np.ndarray) -> np.ndarray:
    return 0.5 * (W + W.T.conj())


def trace_replace(W: np.ndarray, dims: ProcessDims | Sequence[int], systems: Iterable[int]) -> np.ndarray:
    """Apply the replacement map _S W = (I_S/d_S) ⊗ Tr_S(W).

    The implementation uses explicit index loops. It is intentionally simple
    and reliable for the small validation dimensions used in the repository.
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
    """Alias for the process-matrix trace-and-replace map."""
    return trace_replace(W, dims, systems)


def L_valid_bipartite(W: np.ndarray, dims: ProcessDims | Sequence[int]) -> np.ndarray:
    """Project onto the standard bipartite valid-process linear subspace.

    Written in replacement-map notation for subsystem order [AI, AO, BI, BO]:

        L_V(W) = W - _AO W + _AI_AO W - _BO W + _BI_BO W
                 + _AO_BO W - _AI_AO_BO W - _AO_BI_BO W
                 + _AI_AO_BI_BO W.

    This is the convention used by the validation scaffold. Cross-check with
    the target convention before submission.
    """
    rep = lambda S: trace_replace(W, dims, S)
    return (
        W
        - rep([AO])
        + rep([AI, AO])
        - rep([BO])
        + rep([BI, BO])
        + rep([AO, BO])
        - rep([AI, AO, BO])
        - rep([AO, BI, BO])
        + rep([AI, AO, BI, BO])
    )


def L_A_before_B(W: np.ndarray, dims: ProcessDims | Sequence[int]) -> np.ndarray:
    """Baseline projector for the A≺B order subspace.

    The fixed-order subspace is represented as valid bipartite processes with no
    dependence on B's output: W = _BO W.  This scaffold uses _BO∘L_V.
    """
    return trace_replace(L_valid_bipartite(W, dims), dims, [BO])


def L_B_before_A(W: np.ndarray, dims: ProcessDims | Sequence[int]) -> np.ndarray:
    """Baseline projector for the B≺A order subspace.

    The fixed-order subspace is represented as valid bipartite processes with no
    dependence on A's output: W = _AO W.  This scaffold uses _AO∘L_V.
    """
    return trace_replace(L_valid_bipartite(W, dims), dims, [AO])


def white_noise_process(dims: ProcessDims | Sequence[int]) -> np.ndarray:
    """Return a normalized white-noise process with Tr(W)=d_O."""
    dims_tuple = _as_dims(dims)
    D = hilbert_dim(dims_tuple)
    return np.eye(D, dtype=float) * (d_output(dims_tuple) / D)


def random_hermitian(dims: ProcessDims | Sequence[int], seed: int = 0) -> np.ndarray:
    D = hilbert_dim(dims)
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(D, D)) + 1j * rng.normal(size=(D, D))
    return symmetrize(X)


def is_idempotent(projector: Callable[[np.ndarray, ProcessDims | Sequence[int]], np.ndarray], dims: ProcessDims | Sequence[int], *, seed: int = 0, atol: float = 1e-8) -> bool:
    W = random_hermitian(dims, seed=seed)
    P = projector(W, dims)
    return bool(np.allclose(projector(P, dims), P, atol=atol))


def assert_trace_convention(W: np.ndarray, dims: ProcessDims | Sequence[int], *, atol: float = 1e-8) -> None:
    target = d_output(dims)
    if not np.allclose(np.trace(W), target, atol=atol):
        raise AssertionError(f"Trace convention violated: Tr(W)={np.trace(W)} != d_O={target}.")


def superoperator_matrix(projector: Callable[[np.ndarray, ProcessDims | Sequence[int]], np.ndarray], dims: ProcessDims | Sequence[int], *, order: str = "C") -> np.ndarray:
    """Return matrix P such that vec(projector(W)) = P vec(W)."""
    D = hilbert_dim(dims)
    N = D * D
    P = np.zeros((N, N), dtype=float)
    for k in range(N):
        E = np.zeros((D, D), dtype=float)
        if order == "C":
            i, j = divmod(k, D)
        else:
            j, i = divmod(k, D)
        E[i, j] = 1.0
        P[:, k] = np.asarray(projector(E, dims), dtype=float).reshape(-1, order=order)
    return P
