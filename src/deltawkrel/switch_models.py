"""Process-matrix validation targets, including the ideal quantum switch.

Fixed-order and white-noise processes validate the bipartite projectors and the
causally separable SDP wiring.  The ideal quantum switch is implemented in the
convention of Araújo, Branciard, Costa, Feix, Giarmatzi, Brukner,
"Witnessing causal nonseparability", New J. Phys. 17, 102001 (2015):
a rank-one process matrix on [AI, AO, BI, BO, F] with global future
F = F_target ⊗ F_control, target initial state |ψ⟩ and control state |+⟩.
Its generalized robustness (≈ 0.5454) is the external benchmark reproduced by
:func:`deltawkrel.sdp.solve_switch_generalized_robustness`.
"""
from __future__ import annotations

import numpy as np

from .projectors import (
    AI, AO, BI, BO,
    ProcessDims,
    SWITCH_DIMS_WITH_FUTURE,
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


def _switch_branch_vector(order: str, psi: np.ndarray) -> np.ndarray:
    """Pure comb vector |w_{X≺Y}⟩ on [AI, AO, BI, BO, Ft, Fc] (all qubits).

    Convention (Araújo et al., NJP 17, 102001 (2015)):

        |w_{A≺B}⟩ = |ψ⟩_AI |1⟩⟩_{AO,BI} |1⟩⟩_{BO,Ft} |0⟩_Fc,
        |w_{B≺A}⟩ = |ψ⟩_BI |1⟩⟩_{BO,AI} |1⟩⟩_{AO,Ft} |1⟩_Fc,

    with |1⟩⟩ = Σ_j |jj⟩ the unnormalized maximally entangled vector (the
    Choi vector of the identity channel).
    """
    d = 2

    def idx(ai: int, ao: int, bi: int, bo: int, ft: int, fc: int) -> int:
        return ((((ai * d + ao) * d + bi) * d + bo) * d + ft) * d + fc

    w = np.zeros(d ** 6, dtype=float)
    if order == "AB":
        for a in range(d):
            for j in range(d):
                for k in range(d):
                    w[idx(a, j, j, k, k, 0)] += float(psi[a])
    elif order == "BA":
        for b in range(d):
            for j in range(d):
                for k in range(d):
                    w[idx(j, k, b, j, k, 1)] += float(psi[b])
    else:
        raise ValueError("order must be 'AB' or 'BA'.")
    return w


def _validate_switch_target_state(psi: np.ndarray | None) -> np.ndarray:
    if psi is None:
        psi = np.array([1.0, 0.0])
    psi = np.asarray(psi, dtype=float)
    if psi.shape != (2,):
        raise ValueError("psi must be a real qubit state vector of shape (2,).")
    norm = float(np.linalg.norm(psi))
    if norm < 1e-12:
        raise ValueError("psi must be a nonzero state vector.")
    return psi / norm


def ideal_quantum_switch_process(psi: np.ndarray | None = None) -> np.ndarray:
    """Ideal quantum switch process matrix W = |w⟩⟨w| on [AI, AO, BI, BO, F].

    Convention: Araújo, Branciard, Costa, Feix, Giarmatzi, Brukner,
    "Witnessing causal nonseparability", New J. Phys. 17, 102001 (2015).
    All party systems are qubits; the global future F = F_target ⊗ F_control
    has dimension 4 (tensor order: target first, control second); the control
    starts in |+⟩ and the target in |ψ⟩ (default |0⟩, real amplitudes so that
    W is real symmetric):

        |w⟩ = (|w_{A≺B}⟩|0⟩_Fc + |w_{B≺A}⟩|1⟩_Fc)/√2.

    The result is a rank-one PSD matrix of dimension 64 with Tr(W) = d_O = 4.
    Its generalized robustness against causally separable processes is the
    published benchmark value ≈ 0.5454.
    """
    psi = _validate_switch_target_state(psi)
    w = (_switch_branch_vector("AB", psi) + _switch_branch_vector("BA", psi)) / np.sqrt(2.0)
    return np.outer(w, w)


def biased_coherent_switch_process(probability_A_before_B: float, psi: np.ndarray | None = None) -> np.ndarray:
    """Coherent switch with biased order amplitudes.

    ``probability_A_before_B`` is the branch weight q in the pure vector

        |w(q)> = sqrt(q) |w_AB> + sqrt(1-q) |w_BA>.

    The endpoints q=0 and q=1 are fixed-order combs; q=1/2 is the ideal
    balanced switch.  The trace convention remains Tr(W)=4 for all q.
    """
    q = float(probability_A_before_B)
    if not (0.0 <= q <= 1.0):
        raise ValueError("probability_A_before_B must lie in [0, 1].")
    psi = _validate_switch_target_state(psi)
    w = np.sqrt(q) * _switch_branch_vector("AB", psi) + np.sqrt(1.0 - q) * _switch_branch_vector("BA", psi)
    return np.outer(w, w)


def switch_branch_process(order: str, psi: np.ndarray | None = None) -> np.ndarray:
    """Normalized fixed-order branch W_{X≺Y≺F} = |w_{X≺Y}⟩⟨w_{X≺Y}| of the switch.

    Each branch is a valid A≺B≺F (resp. B≺A≺F) comb with Tr(W) = 4; it lies in
    the corresponding fixed-order subspace L_A_before_B_with_future /
    L_B_before_A_with_future.
    """
    psi = _validate_switch_target_state(psi)
    w = _switch_branch_vector(order, psi)
    return np.outer(w, w)


def dephased_switch_process(psi: np.ndarray | None = None) -> np.ndarray:
    """Control-dephased switch: the causally separable mixture of both branches.

    Dephasing the control qubit of the ideal switch removes all coherence
    between the two orders and yields (W_{A≺B≺F} + W_{B≺A≺F})/2, which is
    causally separable by construction (generalized robustness 0).
    """
    psi = _validate_switch_target_state(psi)
    return 0.5 * (switch_branch_process("AB", psi) + switch_branch_process("BA", psi))


def partially_dephased_switch_process(lambda_dephasing: float, psi: np.ndarray | None = None) -> np.ndarray:
    """Switch with partial dephasing of the control qubit.

    ``lambda_dephasing = 0`` gives the ideal coherent switch.  ``lambda_dephasing
    = 1`` gives the fully control-dephased, causally separable branch mixture.
    Intermediate values damp only the off-diagonal order-coherence block:

        W(lambda) = (1 - lambda) W_switch + lambda W_dephased.

    This one-parameter family is the natural falsification curve between the
    published ideal-switch benchmark and the classical fixed-order mixture.
    """
    lam = float(lambda_dephasing)
    if not (0.0 <= lam <= 1.0):
        raise ValueError("lambda_dephasing must lie in [0, 1].")
    return (1.0 - lam) * ideal_quantum_switch_process(psi) + lam * dephased_switch_process(psi)


def switch_white_noise_process(dims: tuple[int, int, int, int, int] = SWITCH_DIMS_WITH_FUTURE) -> np.ndarray:
    """White-noise process with future space and Tr(W)=d_AO*d_BO=4."""
    dims = tuple(int(d) for d in dims)
    if len(dims) != 5 or any(d <= 0 for d in dims):
        raise ValueError("dims must have length 5 with positive entries.")
    D = int(np.prod(dims))
    d_O = int(dims[AO] * dims[BO])
    return np.eye(D, dtype=float) * (d_O / D)


def white_visibility_switch_process(visibility: float, psi: np.ndarray | None = None) -> np.ndarray:
    """Switch mixed with valid white noise.

    ``visibility=1`` is the ideal switch and ``visibility=0`` is the normalized
    white-noise process.  This family is useful for visibility-threshold SDP
    scans.
    """
    v = float(visibility)
    if not (0.0 <= v <= 1.0):
        raise ValueError("visibility must lie in [0, 1].")
    return v * ideal_quantum_switch_process(psi) + (1.0 - v) * switch_white_noise_process()


def white_noise_validation_process(dims: ProcessDims = ProcessDims()) -> np.ndarray:
    return white_noise_process(dims)
