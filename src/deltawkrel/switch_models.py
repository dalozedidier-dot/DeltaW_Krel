"""Small process-matrix validation targets.

The ideal quantum switch is not faked.  This module supplies exact fixed-order
and white-noise processes that are sufficient to validate the projectors and the
causally separable SDP wiring.  The ideal-switch benchmark must still be added
from a selected published convention before a formal submission.
"""
from __future__ import annotations

import numpy as np

from .projectors import (
    AI, AO, BI, BO,
    ProcessDims,
    _as_dims,
    d_output,
    hilbert_dim,
    maximally_entangled_projector,
    white_noise_process,
)


def _kron_all(items: list[np.ndarray]) -> np.ndarray:
    out = np.array([[1.0]])
    for X in items:
        out = np.kron(out, X)
    return out


def fixed_order_A_before_B_process(dims: ProcessDims = ProcessDims()) -> np.ndarray:
    """A simple A≺B process with Tr(W)=d_O.

    W = I_AI/d_AI ⊗ |Φ+><Φ+|_{AO,BI} ⊗ I_BO.
    It represents a maximally mixed input to Alice, identity channel from
    Alice's output to Bob's input, and no future after Bob's output.
    """
    d = _as_dims(dims)
    if d[AO] != d[BI]:
        raise ValueError("AO and BI must have equal dimension for the identity channel.")
    I_AI = np.eye(d[AI]) / d[AI]
    Phi_AO_BI = maximally_entangled_projector(d[AO], d[BI])
    I_BO = np.eye(d[BO])
    # Need tensor order [AI, AO, BI, BO]; Phi already lives on [AO, BI].
    W = _kron_all([I_AI, Phi_AO_BI, I_BO])
    return np.asarray(W, dtype=float)


def fixed_order_B_before_A_process(dims: ProcessDims = ProcessDims()) -> np.ndarray:
    """A simple B≺A process with Tr(W)=d_O.

    W = I_BI/d_BI ⊗ |Φ+><Φ+|_{BO,AI} ⊗ I_AO, reordered to [AI, AO, BI, BO].
    """
    d = _as_dims(dims)
    if d[BO] != d[AI]:
        raise ValueError("BO and AI must have equal dimension for the identity channel.")

    # Build in order [BI, BO, AI, AO], then permute to [AI, AO, BI, BO].
    I_BI = np.eye(d[BI]) / d[BI]
    Phi_BO_AI = maximally_entangled_projector(d[BO], d[AI])
    I_AO = np.eye(d[AO])
    W_src = _kron_all([I_BI, Phi_BO_AI, I_AO])

    src_dims = (d[BI], d[BO], d[AI], d[AO])
    dst_dims = (d[AI], d[AO], d[BI], d[BO])
    D = hilbert_dim(dst_dims)
    # permutation from source factor order [BI,BO,AI,AO] to [AI,AO,BI,BO]
    # source axes index: BI=0, BO=1, AI=2, AO=3; wanted [2,3,0,1]
    W_tensor = W_src.reshape(*src_dims, *src_dims)
    perm = [2, 3, 0, 1, 6, 7, 4, 5]
    W_dst = np.transpose(W_tensor, axes=perm).reshape(D, D)
    return np.asarray(W_dst, dtype=float)


def causally_separable_mixture(dims: ProcessDims = ProcessDims(), q: float = 0.5) -> np.ndarray:
    """Convex mixture q W_A≺B + (1-q) W_B≺A."""
    if not (0.0 <= q <= 1.0):
        raise ValueError("q must lie in [0,1].")
    return q * fixed_order_A_before_B_process(dims) + (1.0 - q) * fixed_order_B_before_A_process(dims)


def ideal_quantum_switch_process(*args, **kwargs):
    """Not implemented by design.

    The ideal quantum switch requires an explicit convention with a future global
    system/control degree of freedom.  It must be implemented from the selected
    literature convention and benchmarked before the repository can be labelled
    submission-ready.
    """
    raise NotImplementedError(
        "ideal_quantum_switch_process is not implemented yet. Complete from the "
        "chosen quantum-switch convention and validate against a published benchmark."
    )


def white_noise_validation_process(dims: ProcessDims = ProcessDims()) -> np.ndarray:
    return white_noise_process(dims)
