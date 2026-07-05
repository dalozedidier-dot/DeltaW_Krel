# Banc numérique linéarisé pour le supplément technique ΔW/K_rel
# Statut : script de contrôle méthodologique, sans numéro de version publié.
# Ce fichier correspond au niveau "toy / banc géométrique" du manuscrit.
# Le SDP causal complet, les projecteurs L_V/L_AB/L_BA et la tomographie réaliste
# relèvent du paquet reproductible séparé et ne doivent pas être confondus avec ce script.

"""
ANNEXE C bis : Banc numérique de contrôle, architecture de calcul corrigée.

Objectif
--------
Ce script teste la chaîne statistique ΔW/K_rel dans un modèle linéarisé. Il
sépare strictement deux niveaux qui ne doivent pas être confondus :

1. Construction de l'espace admissible A :
   A = T_{W0}W ∩ N_cal^⊥ ∩ M0.
   Cette étape contient uniquement des contraintes linéaires : tangent space,
   orthogonalité au bruit calibré et contraintes marginales/non-signalisation.
   Le secteur ICO et les critères de non-séparabilité causale ne sont pas
   inclus dans cette intersection.

2. Construction causale de la direction testée :
   le witness S_opt intervient seulement après construction de A, via
   K_rel = P_A S_opt / ||P_A S_opt||.

Verrou d'applicabilité
----------------------
Avant toute estimation statistique, le script vérifie systématiquement :
- dim(A) > 0 ;
- ||P_A S_opt||_HS > eps_num.
Sinon, la configuration est déclarée inapplicable. Le programme ne renvoie pas
un faux résultat et ne tente pas de normalisation instable.

SDP et robustesse
-----------------
Le Monte Carlo ci-dessous reste un banc linéarisé. Les distances au cône des
processus valides, le calcul de robustesse causale R_rel et le witness optimal
dual doivent être traités par SDP. Le script contient donc un point d'entrée
optionnel pour une vérification SDP régularisée par bruit blanc. Cette partie
requiert cvxpy et, pour une robustesse causale réelle, une paramétrisation
explicite de l'ensemble CS des processus causalement séparables.

Indicateur de transparence
--------------------------
Lorsqu'un calcul SDP régularisé est exécuté, le script reporte :
    omega_white = t_white / (t_white + Σ_i t_i)
Une alerte est levée si omega_white > 0.5, car la robustesse devient alors
dominée par le bruit blanc théorique plutôt que par les bruits instrumentaux
calibrés.

Diagnostics de vigilance
------------------------
Le script peut aussi exécuter un test synthétique de stabilité de la direction
K_rel sous perturbations du witness. Les seuils de soumission courants du manuscrit
sont λ_sens = 0.005 pour la preuve de concept, λ_sens = 0.003 seulement comme
objectif aspirationnel si N_eff devient beaucoup plus grand, λ_ref = 0.001 et
α = 0.01. Les noms ASCII eps_num / eps_R sont réservés aux scripts et JSON.
Ce test ne remplace pas l'analyse de la face optimale du vrai SDP dual ; il sert
de garde-fou numérique pour vérifier
qu'une faible variation d'un witness plausible ne produit pas une direction
projetée complètement différente. Dans l'implémentation complète, ce diagnostic
devra être remplacé par un échantillonnage de la face optimale du dual.
"""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
from scipy.linalg import null_space
from scipy.stats import chi2, norm

try:  # Optionnel : requis seulement pour --sdp-check.
    import cvxpy as cp  # type: ignore
except Exception:  # pragma: no cover - cvxpy peut ne pas être installé.
    cp = None


# ============================================================
# Outils algébriques
# ============================================================


def symmetrize(A: np.ndarray) -> np.ndarray:
    """Symétrise une matrice ou un lot de matrices."""
    return 0.5 * (A + np.swapaxes(A, -1, -2))


def traceless(A: np.ndarray) -> np.ndarray:
    """Retire la composante de trace d'une matrice carrée ou d'un lot de matrices."""
    dim = A.shape[-1]
    eye = np.eye(dim, dtype=A.dtype)
    traces = np.trace(A, axis1=-2, axis2=-1)
    return A - traces[..., None, None] * eye / dim


def normalize_frobenius(A: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """Normalise une matrice selon la norme de Frobenius."""
    norm_A = float(np.linalg.norm(A))
    if norm_A < eps:
        raise ValueError("Impossible de normaliser une matrice de norme quasi nulle.")
    return A / norm_A


def frob_inner(A: np.ndarray, B: np.ndarray) -> float:
    """Produit scalaire de Frobenius <A,B> = Tr(A^T B)."""
    return float(np.sum(A * B))


def make_symmetric_basis(dim: int) -> List[np.ndarray]:
    """Base orthonormée de l'espace des matrices réelles symétriques dim×dim."""
    basis: List[np.ndarray] = []
    for i in range(dim):
        for j in range(i, dim):
            M = np.zeros((dim, dim), dtype=float)
            if i == j:
                M[i, i] = 1.0
            else:
                M[i, j] = M[j, i] = 1.0 / np.sqrt(2.0)
            basis.append(M)
    return basis


def project_onto_basis(M: np.ndarray, basis: Sequence[np.ndarray]) -> np.ndarray:
    """Projette M sur une base orthonormée au sens de Frobenius."""
    proj = np.zeros_like(M, dtype=float)
    for B in basis:
        proj += frob_inner(M, B) * B
    return proj


def gram_schmidt_add(
    candidate: np.ndarray,
    basis: List[np.ndarray],
    eps: float = 1e-10,
) -> Optional[np.ndarray]:
    """Ajoute une matrice à une base orthonormée après orthogonalisation."""
    X = candidate.copy()
    for B in basis:
        X = X - frob_inner(X, B) * B
    norm_X = float(np.linalg.norm(X))
    if norm_X < eps:
        return None
    return X / norm_X


# ============================================================
# Contraintes M0 toy : marges nulles optionnelles
# ============================================================


def partial_trace_A(M: np.ndarray, dA: int, dB: int) -> np.ndarray:
    """Trace partielle sur A pour une matrice sur H_A ⊗ H_B."""
    T = M.reshape(dA, dB, dA, dB)
    return np.einsum("iajb->ab", T)


def partial_trace_B(M: np.ndarray, dA: int, dB: int) -> np.ndarray:
    """Trace partielle sur B pour une matrice sur H_A ⊗ H_B."""
    T = M.reshape(dA, dB, dA, dB)
    return np.einsum("iajb->ij", T)


def marginal_constraint_matrices(dim: int, subsystem_dim: Optional[int] = None) -> List[np.ndarray]:
    """
    Construit des contraintes linéaires toy de marges nulles.

    On interprète l'espace de dimension dim comme H_A ⊗ H_B avec dA*dB=dim.
    Si subsystem_dim est None, on prend dA=dB=sqrt(dim) lorsque c'est possible.
    Les contraintes imposent Tr_A(M)=0 et Tr_B(M)=0, sous forme de matrices C telles
    que <C, M> = 0.
    """
    if subsystem_dim is None:
        # Factorisation automatique dim = dA * dB avec dA le plus proche de sqrt(dim).
        dA = None
        for cand in range(int(np.sqrt(dim)), 1, -1):
            if dim % cand == 0:
                dA = cand
                break
        if dA is None:
            raise ValueError(
                "include_M0=True exige un dim composite ou subsystem_dim explicite."
            )
        dB = dim // dA
    else:
        dA = int(subsystem_dim)
        if dim % dA != 0:
            raise ValueError("subsystem_dim doit diviser dim.")
        dB = dim // dA

    constraints: List[np.ndarray] = []

    # Tr_A(M)_{a,b} = Σ_i M[(i,a),(i,b)]
    for a in range(dB):
        for b in range(dB):
            C = np.zeros((dim, dim), dtype=float)
            for i in range(dA):
                row = i * dB + a
                col = i * dB + b
                C[row, col] += 1.0
            constraints.append(symmetrize(C))

    # Tr_B(M)_{i,j} = Σ_a M[(i,a),(j,a)]
    for i in range(dA):
        for j in range(dA):
            C = np.zeros((dim, dim), dtype=float)
            for a in range(dB):
                row = i * dB + a
                col = j * dB + a
                C[row, col] += 1.0
            constraints.append(symmetrize(C))

    return constraints


def build_admissible_space(
    dim: int,
    N_cal_basis: Sequence[np.ndarray],
    impose_trace_zero: bool = True,
    include_M0: bool = False,
    subsystem_dim: Optional[int] = None,
    rcond: float = 1e-10,
) -> List[np.ndarray]:
    """
    Construit une base orthonormée de l'espace admissible A.

    Modèle toy :
      T_{W0}W : matrices symétriques de trace nulle ;
      N_cal^⊥ : orthogonalité à toutes les directions de bruit calibré ;
      M0 : optionnel, contraintes de marges nulles via traces partielles.

    Important : aucun critère ICO, aucune distance à CS et aucun witness causal
    n'est introduit ici. A est uniquement une intersection de sous-espaces
    linéaires. Les propriétés causales interviennent ensuite, par projection du
    witness sur A.

    La base est construite comme le noyau d'une matrice de contraintes linéaires
    exprimées dans la base orthonormée des matrices symétriques.
    """
    sym_basis = make_symmetric_basis(dim)

    constraint_mats: List[np.ndarray] = []
    if impose_trace_zero:
        constraint_mats.append(np.eye(dim))
    constraint_mats.extend(N_cal_basis)

    if include_M0:
        constraint_mats.extend(marginal_constraint_matrices(dim, subsystem_dim=subsystem_dim))

    if not constraint_mats:
        return sym_basis

    coeff_rows = []
    for C in constraint_mats:
        coeff_rows.append([frob_inner(C, B) for B in sym_basis])

    A_eq = np.asarray(coeff_rows, dtype=float)
    ns = null_space(A_eq, rcond=rcond)  # colonnes = base du noyau

    A_basis: List[np.ndarray] = []
    for col in range(ns.shape[1]):
        coeffs = ns[:, col]
        M = np.zeros((dim, dim), dtype=float)
        for coeff, B in zip(coeffs, sym_basis):
            M += coeff * B
        # Sécurité numérique : projection exacte sur symétrique et trace nulle
        M = symmetrize(M)
        norm_M = float(np.linalg.norm(M))
        if norm_M > 1e-12:
            A_basis.append(M / norm_M)

    return A_basis


# ============================================================
# Structures de données
# ============================================================


class InapplicableConfiguration(RuntimeError):
    """Configuration où aucune direction relationnelle testable ne peut être définie."""


@dataclass
class Basis:
    """Base toy du banc de contrôle."""

    W_QM: np.ndarray
    W_0: np.ndarray
    N_cal_basis: List[np.ndarray]
    S_opt: np.ndarray
    K_rel: np.ndarray
    A_basis: List[np.ndarray]
    A_dim: int
    projection_norm: float
    witness_coupling: float
    eps_num: float
    applicable: bool = True


@dataclass
class RobustnessDiagnostics:
    """Diagnostics de robustesse SDP régularisée par bruit blanc."""

    status: str
    objective_value: float
    t_white: float
    t_calibrated_sum: float
    omega_white: float
    omega_white_alert: bool
    white_alert_threshold: float
    solver: str
    note: str


@dataclass
class ThresholdInfo:
    """Seuil LR et distribution nulle éventuelle."""

    threshold: float
    null_lr: Optional[np.ndarray] = None


@dataclass
class ResultRow:
    """Ligne de résultats exportée en CSV/JSON."""

    lambda_true: float
    n_samples: int
    power: float
    power_ci_low: float
    power_ci_high: float
    detections: int
    n_sim: int
    lambda_fit_mean: float
    lambda_fit_std: float
    lambda_fit_q05: float
    lambda_fit_q50: float
    lambda_fit_q95: float
    beta_fit_norm_mean: float
    beta_fit_norm_std: float
    LR_mean: float
    LR_q50: float
    LR_q95: float
    p_value_mean: float
    LR_threshold: float
    alpha_level: float
    sigma_obs: float
    alpha_true: float
    dim: int
    n_noise: int
    A_dim: int
    projection_norm: float
    witness_coupling: float
    max_inner_N_K_rel: float
    proj_norm_relative_to_random: float
    eps_num: float
    omega_white: float
    omega_white_alert: bool
    sdp_status: str
    lambda_positive_only: bool
    threshold_mode: str
    include_M0: bool


# ============================================================
# Statistiques
# ============================================================


def wilson_ci(k: int, n: int, confidence: float = 0.95) -> Tuple[float, float]:
    """Intervalle de confiance de Wilson pour une proportion."""
    if n <= 0:
        return float("nan"), float("nan")
    z = float(norm.ppf(1.0 - (1.0 - confidence) / 2.0))
    p = k / n
    denominator = 1.0 + z**2 / n
    center = (p + z**2 / (2.0 * n)) / denominator
    half_width = z * np.sqrt((p * (1.0 - p) + z**2 / (4.0 * n)) / n) / denominator
    return max(0.0, float(center - half_width)), min(1.0, float(center + half_width))


def asymptotic_lr_threshold(alpha_level: float, lambda_positive_only: bool) -> float:
    """Seuil asymptotique du LR, avec mélange frontière si λ≥0."""
    if not (0.0 < alpha_level < 1.0):
        raise ValueError("alpha_level doit être strictement compris entre 0 et 1.")
    if lambda_positive_only:
        if 2.0 * alpha_level >= 1.0:
            raise ValueError("Avec lambda_positive_only=True, alpha_level doit être < 0.5.")
        return float(chi2.isf(2.0 * alpha_level, df=1))
    return float(chi2.isf(alpha_level, df=1))


def asymptotic_p_values(LR: np.ndarray, lambda_positive_only: bool) -> np.ndarray:
    """p-values asymptotiques du test LR."""
    LR = np.asarray(LR, dtype=float)
    if lambda_positive_only:
        return np.where(LR <= 1e-12, 1.0, 0.5 * chi2.sf(LR, df=1))
    return chi2.sf(LR, df=1)


def empirical_p_values(LR: np.ndarray, null_lr: np.ndarray) -> np.ndarray:
    """p-values empiriques avec correction +1."""
    LR = np.asarray(LR, dtype=float)
    null_lr = np.sort(np.asarray(null_lr, dtype=float))
    n_null = len(null_lr)
    if n_null <= 0:
        raise ValueError("La distribution nulle empirique est vide.")
    left_counts = np.searchsorted(null_lr, LR, side="left")
    ge_counts = n_null - left_counts
    return (ge_counts + 1.0) / (n_null + 1.0)


# ============================================================
# Construction du modèle de contrôle
# ============================================================


def random_traceless_symmetric(dim: int, rng: np.random.Generator) -> np.ndarray:
    """Tire une direction symétrique, sans trace, normalisée."""
    M = symmetrize(rng.normal(0.0, 1.0, size=(dim, dim)))
    M = traceless(M)
    return normalize_frobenius(M)


def build_noise_basis(dim: int, n_noise: int, rng: np.random.Generator) -> List[np.ndarray]:
    """Construit n_noise directions de bruit calibré orthonormées."""
    if n_noise < 1:
        raise ValueError("n_noise doit être >= 1.")
    max_dim = dim * (dim + 1) // 2 - 1
    if n_noise >= max_dim:
        raise ValueError(f"n_noise trop grand pour l'espace traceless symétrique (max {max_dim-1}).")

    basis: List[np.ndarray] = []
    attempts = 0
    while len(basis) < n_noise:
        attempts += 1
        if attempts > 1000:
            raise RuntimeError("Impossible de construire une base de bruit orthonormée.")
        candidate = random_traceless_symmetric(dim, rng)
        ortho = gram_schmidt_add(candidate, basis)
        if ortho is not None:
            basis.append(ortho)
    return basis


def apply_applicability_lock(
    A_basis: Sequence[np.ndarray],
    S_opt: np.ndarray,
    eps_num: float,
) -> Tuple[np.ndarray, float]:
    """
    Applique le verrou d'applicabilité numérique.

    La statistique n'est autorisée que si A est non vide et si la projection du
    witness sur A dépasse le seuil eps_num. Cette fonction centralise la garde
    afin d'éviter toute estimation fondée sur une direction quasi nulle.
    """
    if len(A_basis) <= 0:
        raise InapplicableConfiguration("dim(A)=0 : espace admissible vide.")

    projected = project_onto_basis(S_opt, A_basis)
    projection_norm = float(np.linalg.norm(projected))
    if projection_norm <= eps_num:
        raise InapplicableConfiguration(
            f"||P_A S_opt|| = {projection_norm:.3e} <= eps_num = {eps_num:.1e}."
        )
    return projected, projection_norm


def build_basis_control(
    dim: int,
    rng: np.random.Generator,
    alpha_true: float,
    n_noise: int = 3,
    include_M0: bool = False,
    subsystem_dim: Optional[int] = None,
    eps_num: float = 1e-8,
) -> Basis:
    """Construit W_QM, W0, N_cal, witness synthétique et K_rel projeté."""
    W_QM = np.eye(dim) / dim
    perturb = random_traceless_symmetric(dim, rng)
    W_QM = W_QM + 0.01 * perturb
    W_QM = W_QM / np.trace(W_QM)

    N_cal_basis = build_noise_basis(dim=dim, n_noise=n_noise, rng=rng)
    W_0 = W_QM + alpha_true * N_cal_basis[0]

    # Witness synthétique : direction non choisie d'après les données.
    # La projection sur A fera le retrait des composantes de bruit/marges.
    S_opt = random_traceless_symmetric(dim, rng)

    A_basis = build_admissible_space(
        dim=dim,
        N_cal_basis=N_cal_basis,
        impose_trace_zero=True,
        include_M0=include_M0,
        subsystem_dim=subsystem_dim,
    )
    projected, projection_norm = apply_applicability_lock(
        A_basis=A_basis,
        S_opt=S_opt,
        eps_num=eps_num,
    )

    K_rel = projected / projection_norm
    witness_coupling = float(frob_inner(S_opt, K_rel))

    return Basis(
        W_QM=W_QM,
        W_0=W_0,
        N_cal_basis=N_cal_basis,
        S_opt=S_opt,
        K_rel=K_rel,
        A_basis=list(A_basis),
        A_dim=len(A_basis),
        projection_norm=projection_norm,
        witness_coupling=witness_coupling,
        eps_num=float(eps_num),
        applicable=True,
    )


def basis_diagnostics(basis: Basis) -> Dict[str, float]:
    """Diagnostics numériques de la base."""
    dim = basis.W_QM.shape[0]
    D_traceless = dim * (dim + 1) // 2 - 1
    inners = np.array([frob_inner(basis.K_rel, N) for N in basis.N_cal_basis])
    expected_random = float(np.sqrt(basis.A_dim / D_traceless))
    out: Dict[str, float] = {
        "trace_W_QM": float(np.trace(basis.W_QM)),
        "trace_W_0": float(np.trace(basis.W_0)),
        "trace_K_rel": float(np.trace(basis.K_rel)),
        "norm_K_rel": float(np.linalg.norm(basis.K_rel)),
        "A_dim": float(basis.A_dim),
        "projection_norm": float(basis.projection_norm),
        "witness_coupling": float(basis.witness_coupling),
        "eps_num": float(basis.eps_num),
        "applicable": float(1.0 if basis.applicable else 0.0),
        # Diagnostics renforcés : orthogonalité de K_rel au bruit calibré.
        "max_inner_N_K_rel": float(np.max(np.abs(inners))),
        "norm_proj_K_rel_on_N_cal": float(np.linalg.norm(inners)),
        # Indicateur de typicalité : pour une direction aléatoire normalisée dans
        # l'espace symétrique sans trace de dimension D, E||P_A S||^2 = A_dim / D.
        # Un ratio très inférieur à 1 signale une direction structurellement fragile.
        "expected_proj_norm_random": expected_random,
        "proj_norm_relative_to_random": float(basis.projection_norm / expected_random),
    }
    for i, N in enumerate(basis.N_cal_basis):
        out[f"trace_N_cal_{i}"] = float(np.trace(N))
        out[f"norm_N_cal_{i}"] = float(np.linalg.norm(N))
        out[f"inner_K_N_cal_{i}"] = float(frob_inner(basis.K_rel, N))
        for j in range(i):
            out[f"inner_N_cal_{i}_{j}"] = float(frob_inner(N, basis.N_cal_basis[j]))
    return out


def witness_stability_diagnostics(
    basis: Basis,
    rng: np.random.Generator,
    n_samples: int = 200,
    perturb_scale: float = 0.05,
    cos_threshold: float = 0.90,
    eps_num: Optional[float] = None,
) -> Dict[str, float]:
    """
    Diagnostic synthétique de stabilité de K_rel sous variations du witness.

    Attention : dans le vrai SDP causal, il faut échantillonner la face optimale
    du dual. Ici, on teste seulement la sensibilité numérique de la projection
    P_A S_opt dans le banc toy, en perturbant S_opt par des directions
    symétriques sans trace. Une instabilité forte signale que la direction
    projetée est fragile et qu'une règle de sélection du witness doit être
    pré-enregistrée avant toute interprétation confirmatoire.
    """
    if eps_num is None:
        eps_num = basis.eps_num
    if n_samples <= 0:
        return {
            "witness_stability_run": 0.0,
            "n_witness_samples": 0.0,
            "witness_rejected_samples": 0.0,
            "witness_perturb_scale": float(perturb_scale),
            "witness_cos_min": float("nan"),
            "witness_cos_mean": float("nan"),
            "witness_cos_q05": float("nan"),
            "witness_cos_q50": float("nan"),
            "witness_cos_q95": float("nan"),
            "witness_unstable_alert": 0.0,
            "witness_cos_threshold": float(cos_threshold),
        }

    cosines: List[float] = []
    rejected = 0
    for _ in range(int(n_samples)):
        perturb = random_traceless_symmetric(basis.W_QM.shape[0], rng)
        candidate = basis.S_opt + float(perturb_scale) * perturb
        try:
            projected, norm_projected = apply_applicability_lock(
                basis.A_basis,
                candidate,
                eps_num=float(eps_num),
            )
        except InapplicableConfiguration:
            rejected += 1
            continue
        k_candidate = projected / norm_projected
        cos = frob_inner(basis.K_rel, k_candidate)
        cosines.append(float(abs(cos)))

    if not cosines:
        return {
            "witness_stability_run": 1.0,
            "n_witness_samples": float(n_samples),
            "witness_rejected_samples": float(rejected),
            "witness_perturb_scale": float(perturb_scale),
            "witness_cos_min": float("nan"),
            "witness_cos_mean": float("nan"),
            "witness_cos_q05": float("nan"),
            "witness_cos_q50": float("nan"),
            "witness_cos_q95": float("nan"),
            "witness_unstable_alert": 1.0,
            "witness_cos_threshold": float(cos_threshold),
        }

    values = np.asarray(cosines, dtype=float)
    return {
        "witness_stability_run": 1.0,
        "n_witness_samples": float(n_samples),
        "witness_rejected_samples": float(rejected),
        "witness_perturb_scale": float(perturb_scale),
        "witness_cos_min": float(np.min(values)),
        "witness_cos_mean": float(np.mean(values)),
        "witness_cos_q05": float(np.quantile(values, 0.05)),
        "witness_cos_q50": float(np.quantile(values, 0.50)),
        "witness_cos_q95": float(np.quantile(values, 0.95)),
        "witness_unstable_alert": float(np.min(values) < float(cos_threshold)),
        "witness_cos_threshold": float(cos_threshold),
    }


# ============================================================
# Simulation et estimation
# ============================================================


def simulate_observations(
    W_true: np.ndarray,
    n_samples: int,
    n_sim: int,
    sigma_obs: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Simule l'estimateur moyen observé de W avec bruit gaussien symétrisé."""
    if n_samples <= 0:
        raise ValueError("n_samples doit être > 0.")
    if n_sim <= 0:
        raise ValueError("n_sim doit être > 0.")
    if sigma_obs <= 0:
        raise ValueError("sigma_obs doit être > 0.")

    dim = W_true.shape[0]
    noise = rng.normal(
        loc=0.0,
        scale=sigma_obs / np.sqrt(n_samples),
        size=(n_sim, dim, dim),
    )

    W_obs = W_true[None, :, :] + symmetrize(noise)

    # Recentrage exact de la trace à 1, puis symétrisation finale par sécurité
    # numérique dans les simulations massives.
    W_obs = W_obs - (np.trace(W_obs, axis1=-2, axis2=-1) - 1.0)[..., None, None] * np.eye(dim) / dim
    W_obs = symmetrize(W_obs)
    return W_obs


def estimate_lr_vectorized(
    W_obs: np.ndarray,
    basis: Basis,
    n_samples: int,
    sigma_obs: float,
    lambda_positive_only: bool = True,
) -> Dict[str, np.ndarray]:
    """
    Estime β_i et λ par projections dans le modèle linéaire.

    Comme K_rel ∈ N_cal^⊥, le LR associé à λ se réduit au gain de projection
    sur K_rel, après retrait du processus nominal W0. Les coefficients β_i
    sont reportés comme diagnostics, mais n'entrent pas dans le LR si la base
    est orthogonale à K_rel.
    """
    if W_obs.ndim != 3:
        raise ValueError("W_obs doit être un tableau (n_sim, dim, dim).")

    residual = W_obs - basis.W_0[None, :, :]

    beta_hats = np.stack(
        [np.einsum("sij,ij->s", residual, N) for N in basis.N_cal_basis],
        axis=1,
    )
    lambda_raw = np.einsum("sij,ij->s", residual, basis.K_rel)

    if lambda_positive_only:
        lambda_hat = np.maximum(lambda_raw, 0.0)
    else:
        lambda_hat = lambda_raw

    LR = n_samples * lambda_hat**2 / (sigma_obs**2)
    LR = np.maximum(LR, 0.0)

    return {
        "beta_hats": beta_hats,
        "lambda_hat": lambda_hat,
        "lambda_raw": lambda_raw,
        "LR": LR,
    }


def calibrate_thresholds(
    basis: Basis,
    n_samples_values: Sequence[int],
    sigma_obs: float,
    alpha_level: float,
    lambda_positive_only: bool,
    threshold_mode: str,
    n_null: int,
    rng: np.random.Generator,
) -> Dict[int, ThresholdInfo]:
    """Calibre les seuils LR pour chaque taille d'échantillon."""
    if threshold_mode not in {"asymptotic", "empirical"}:
        raise ValueError("threshold_mode doit valoir 'asymptotic' ou 'empirical'.")

    thresholds: Dict[int, ThresholdInfo] = {}
    for n_samples in n_samples_values:
        n = int(n_samples)

        if threshold_mode == "asymptotic":
            threshold = asymptotic_lr_threshold(alpha_level, lambda_positive_only)
            thresholds[n] = ThresholdInfo(threshold=threshold, null_lr=None)
            continue

        W_null = basis.W_0
        W_obs_null = simulate_observations(
            W_true=W_null,
            n_samples=n,
            n_sim=int(n_null),
            sigma_obs=sigma_obs,
            rng=rng,
        )
        est_null = estimate_lr_vectorized(
            W_obs=W_obs_null,
            basis=basis,
            n_samples=n,
            sigma_obs=sigma_obs,
            lambda_positive_only=lambda_positive_only,
        )
        null_lr = np.asarray(est_null["LR"], dtype=float)
        threshold = float(np.quantile(null_lr, 1.0 - alpha_level))
        thresholds[n] = ThresholdInfo(threshold=threshold, null_lr=null_lr)

    return thresholds



# ============================================================
# SDP : robustesse régularisée et indicateur omega_white
# ============================================================


def make_white_noise_process(dim: int) -> np.ndarray:
    """Bruit blanc admissible toy, normalisé à trace 1."""
    return np.eye(dim, dtype=float) / dim


def not_run_robustness_diagnostics() -> RobustnessDiagnostics:
    """Diagnostic par défaut lorsque le contrôle SDP n'est pas demandé."""
    return RobustnessDiagnostics(
        status="not_run",
        objective_value=float("nan"),
        t_white=float("nan"),
        t_calibrated_sum=float("nan"),
        omega_white=float("nan"),
        omega_white_alert=False,
        white_alert_threshold=0.5,
        solver="none",
        note="Contrôle SDP non exécuté ; le Monte Carlo reste linéarisé.",
    )


def solve_regularized_sdp_proxy(
    W: np.ndarray,
    N_cal_basis: Sequence[np.ndarray],
    W_white: Optional[np.ndarray] = None,
    white_alert_threshold: float = 0.5,
    solver: Optional[str] = None,
) -> RobustnessDiagnostics:
    """
    Point d'entrée SDP régularisé par bruit blanc.

    Cette fonction est volontairement conservatrice. Elle résout seulement une
    vérification SDP proxy de positivité :

        W + Σ_i t_i N_i + t_white W_white >= 0,
        t_i >= 0, t_white >= 0,
        min Σ_i t_i + t_white.

    Elle ne calcule pas encore la robustesse causale R_rel au sens strict, car
    cela exige une paramétrisation explicite de CS et l'ajout des contraintes SDP
    correspondantes. Pour la soumission, cette validation doit être fournie par
    projectors_definitions.ipynb et validation_switch_ideal.ipynb dans le paquet reproductible. Dans une implémentation expérimentale, remplacer la
    contrainte PSD proxy par les contraintes exactes de CS et extraire le witness
    dual du problème. Avec certains paramètres nominaux, W peut déjà être PSD :
    le contrôle renvoie alors un objectif nul, ce qui est un cas trivialement
    faisable et non une démonstration de robustesse causale.
    """
    if cp is None:
        raise RuntimeError(
            "cvxpy n'est pas installé : impossible d'exécuter --sdp-check dans cet environnement. "
            "Installez cvxpy pour activer le contrôle SDP."
        )

    W = symmetrize(np.asarray(W, dtype=float))
    dim = W.shape[0]
    if W_white is None:
        W_white = make_white_noise_process(dim)
    W_white = symmetrize(np.asarray(W_white, dtype=float))

    k = len(N_cal_basis)
    t_cal = cp.Variable(k, nonneg=True, name="t_cal")
    t_white = cp.Variable(nonneg=True, name="t_white")

    X = W + t_white * W_white
    for i, N_i in enumerate(N_cal_basis):
        X = X + t_cal[i] * symmetrize(np.asarray(N_i, dtype=float))

    constraints = [X >> 0]
    objective = cp.Minimize(cp.sum(t_cal) + t_white)
    problem = cp.Problem(objective, constraints)

    chosen_solver = solver or "CLARABEL"
    requested_solver = chosen_solver
    try:
        problem.solve(solver=chosen_solver, verbose=False)
    except Exception as exc:
        # Fallback fréquent lorsque CLARABEL n'est pas disponible.
        chosen_solver = "SCS"
        print(
            f"AVERTISSEMENT : fallback du solveur {requested_solver} vers {chosen_solver} "
            f"pour le proxy SDP ({type(exc).__name__}: {exc})."
        )
        problem.solve(solver=chosen_solver, verbose=False)

    status = str(problem.status)
    if status not in {"optimal", "optimal_inaccurate"}:
        return RobustnessDiagnostics(
            status=status,
            objective_value=float("nan"),
            t_white=float("nan"),
            t_calibrated_sum=float("nan"),
            omega_white=float("nan"),
            omega_white_alert=False,
            white_alert_threshold=float(white_alert_threshold),
            solver=chosen_solver,
            note="SDP proxy non résolu à l'optimalité ; ne pas interpréter la robustesse.",
        )

    t_white_value = max(0.0, float(t_white.value))
    t_cal_values = np.maximum(0.0, np.asarray(t_cal.value, dtype=float)) if k else np.zeros(0)
    t_cal_sum = float(np.sum(t_cal_values))
    denom = t_white_value + t_cal_sum
    omega_white = float(t_white_value / denom) if denom > 1e-15 else 0.0
    alert = bool(omega_white > white_alert_threshold)
    note = (
        "Alerte : robustesse dominée par le bruit blanc théorique."
        if alert
        else "Bruit blanc non dominant dans le SDP proxy."
    )
    note += " Ce diagnostic est un proxy PSD, pas encore R_rel sur CS."

    return RobustnessDiagnostics(
        status=status,
        objective_value=float(problem.value),
        t_white=t_white_value,
        t_calibrated_sum=t_cal_sum,
        omega_white=omega_white,
        omega_white_alert=alert,
        white_alert_threshold=float(white_alert_threshold),
        solver=chosen_solver,
        note=note,
    )


# ============================================================
# Simulation principale
# ============================================================


def run_monte_carlo_control(
    lambda_true_values: Optional[Sequence[float]] = None,
    n_samples_values: Optional[Sequence[int]] = None,
    n_sim: int = 5000,
    n_null: int = 20000,
    dim: int = 16,
    n_noise: int = 3,
    seed: int = 42,
    seed_basis: Optional[int] = None,
    seed_sim: Optional[int] = None,
    sigma_obs: float = np.sqrt(0.0008),
    alpha_true: float = 0.12,
    alpha_level: float = 0.01,
    lambda_positive_only: bool = True,
    threshold_mode: str = "asymptotic",
    include_M0: bool = False,
    subsystem_dim: Optional[int] = None,
    eps_num: float = 1e-8,
    output_dir: str = "monte_carlo_outputs_control",
    save_outputs: bool = True,
    make_plots: bool = True,
    sdp_check: bool = False,
    white_alert_threshold: float = 0.5,
    sdp_solver: Optional[str] = None,
    witness_stability: bool = False,
    n_witness_samples: int = 200,
    witness_perturb_scale: float = 0.05,
    witness_stability_cos_threshold: float = 0.90,
) -> List[ResultRow]:
    """Lance la simulation Monte Carlo du banc de contrôle."""
    if lambda_true_values is None:
        # 0.003 est conservé comme objectif aspirationnel ; 0.005 est le seuil
        # réaliste de preuve de concept utilisé dans le manuscrit de soumission.
        lambda_true_values = [0.0, 0.003, 0.005, 0.010]
    if n_samples_values is None:
        n_samples_values = [1000, 2000, 5000, 10000, 20000, 50000, 100000]

    lambda_true_values = [float(x) for x in lambda_true_values]
    n_samples_values = [int(x) for x in n_samples_values]

    if seed_basis is None:
        seed_basis = int(seed)
    if seed_sim is None:
        # Décalage fixe pour éviter de réutiliser exactement le même flux aléatoire
        # pour la géométrie du banc et pour les observations simulées.
        seed_sim = int(seed) + 1_000_003

    rng_basis = np.random.default_rng(int(seed_basis))
    rng_sim = np.random.default_rng(int(seed_sim))
    rng_diagnostics = np.random.default_rng(int(seed_sim) + 17)

    try:
        basis = build_basis_control(
            dim=dim,
            rng=rng_basis,
            alpha_true=alpha_true,
            n_noise=n_noise,
            include_M0=include_M0,
            subsystem_dim=subsystem_dim,
            eps_num=eps_num,
        )
    except InapplicableConfiguration as exc:
        # Grille du Tableau 1 : cas d'inapplicabilité.
        print(f"TEST INAPPLICABLE : {exc}")
        print(
            "Le protocole ne définit pas de direction relationnelle testable ; "
            "il ne réfute pas l'hypothèse, il la rend inapplicable dans ce dispositif."
        )
        return []

    diag = basis_diagnostics(basis)
    print(
        "Diagnostics du banc de contrôle : "
        f"A_dim={basis.A_dim} | ||P_A S_opt||={basis.projection_norm:.4f} | "
        f"typicalité={diag['proj_norm_relative_to_random']:.3f} | "
        f"max|<N_i,K_rel>|={diag['max_inner_N_K_rel']:.2e}"
    )
    if diag["proj_norm_relative_to_random"] < 0.5:
        print(
            "ALERTE typicalité : ||P_A S_opt|| est inférieure à 50 % de la projection "
            "attendue pour une direction aléatoire comparable. Ce n'est pas un critère "
            "de rejet, mais un signal de fragilité directionnelle à reporter."
        )

    witness_diag: Dict[str, float] = {"witness_stability_run": 0.0}
    if witness_stability:
        witness_diag = witness_stability_diagnostics(
            basis=basis,
            rng=rng_diagnostics,
            n_samples=int(n_witness_samples),
            perturb_scale=float(witness_perturb_scale),
            cos_threshold=float(witness_stability_cos_threshold),
            eps_num=float(eps_num),
        )
        print(
            "Diagnostic stabilité witness/K_rel : "
            f"cos_min={witness_diag['witness_cos_min']:.3f} | "
            f"cos_moyen={witness_diag['witness_cos_mean']:.3f} | "
            f"alerte={bool(witness_diag['witness_unstable_alert'])}"
        )
        if bool(witness_diag['witness_unstable_alert']):
            print(
                "ALERTE witness : la projection de witnesses proches produit des directions K_rel "
                "trop variables. Dans le SDP complet, échantillonner la face optimale du dual "
                "ou déclarer la direction insuffisamment stable."
            )

    sdp_diag = not_run_robustness_diagnostics()
    if sdp_check:
        sdp_diag = solve_regularized_sdp_proxy(
            W=basis.W_0,
            N_cal_basis=basis.N_cal_basis,
            W_white=make_white_noise_process(dim),
            white_alert_threshold=white_alert_threshold,
            solver=sdp_solver,
        )
        print(
            "Diagnostic SDP : "
            f"status={sdp_diag.status} | "
            f"omega_white={sdp_diag.omega_white:.3f} | "
            f"alerte={sdp_diag.omega_white_alert}"
        )
        if sdp_diag.omega_white_alert:
            print(
                "ALERTE omega_white : la régularisation est dominée par le bruit blanc ; "
                "la robustesse ne doit pas être interprétée comme spécifique au bruit instrumental."
            )

    output_path = Path(output_dir)
    if save_outputs:
        output_path.mkdir(parents=True, exist_ok=True)

    thresholds = calibrate_thresholds(
        basis=basis,
        n_samples_values=n_samples_values,
        sigma_obs=sigma_obs,
        alpha_level=alpha_level,
        lambda_positive_only=lambda_positive_only,
        threshold_mode=threshold_mode,
        n_null=n_null,
        rng=rng_sim,
    )

    results: List[ResultRow] = []
    for lambda_true in lambda_true_values:
        for n_samples in n_samples_values:
            W_true = basis.W_0 + lambda_true * basis.K_rel
            W_obs = simulate_observations(
                W_true=W_true,
                n_samples=n_samples,
                n_sim=int(n_sim),
                sigma_obs=sigma_obs,
                rng=rng_sim,
            )
            est = estimate_lr_vectorized(
                W_obs=W_obs,
                basis=basis,
                n_samples=n_samples,
                sigma_obs=sigma_obs,
                lambda_positive_only=lambda_positive_only,
            )

            LR = np.asarray(est["LR"], dtype=float)
            lambda_hat = np.asarray(est["lambda_hat"], dtype=float)
            beta_hats = np.asarray(est["beta_hats"], dtype=float)

            threshold_info = thresholds[n_samples]
            LR_threshold = float(threshold_info.threshold)

            if threshold_mode == "empirical":
                if threshold_info.null_lr is None:
                    raise RuntimeError("Distribution nulle manquante en mode empirical.")
                p_values = empirical_p_values(LR, threshold_info.null_lr)
            else:
                p_values = asymptotic_p_values(LR, lambda_positive_only)

            detections = int(np.sum(LR > LR_threshold))
            power = detections / int(n_sim)
            ci_low, ci_high = wilson_ci(detections, int(n_sim), confidence=0.95)
            beta_norm = np.linalg.norm(beta_hats, axis=1)

            row = ResultRow(
                lambda_true=float(lambda_true),
                n_samples=int(n_samples),
                power=float(power),
                power_ci_low=float(ci_low),
                power_ci_high=float(ci_high),
                detections=detections,
                n_sim=int(n_sim),
                lambda_fit_mean=float(np.mean(lambda_hat)),
                lambda_fit_std=float(np.std(lambda_hat)),
                lambda_fit_q05=float(np.quantile(lambda_hat, 0.05)),
                lambda_fit_q50=float(np.quantile(lambda_hat, 0.50)),
                lambda_fit_q95=float(np.quantile(lambda_hat, 0.95)),
                beta_fit_norm_mean=float(np.mean(beta_norm)),
                beta_fit_norm_std=float(np.std(beta_norm)),
                LR_mean=float(np.mean(LR)),
                LR_q50=float(np.quantile(LR, 0.50)),
                LR_q95=float(np.quantile(LR, 0.95)),
                p_value_mean=float(np.mean(p_values)),
                LR_threshold=float(LR_threshold),
                alpha_level=float(alpha_level),
                sigma_obs=float(sigma_obs),
                alpha_true=float(alpha_true),
                dim=int(dim),
                n_noise=int(n_noise),
                A_dim=int(basis.A_dim),
                projection_norm=float(basis.projection_norm),
                witness_coupling=float(basis.witness_coupling),
                max_inner_N_K_rel=float(diag["max_inner_N_K_rel"]),
                proj_norm_relative_to_random=float(diag["proj_norm_relative_to_random"]),
                eps_num=float(eps_num),
                omega_white=float(sdp_diag.omega_white),
                omega_white_alert=bool(sdp_diag.omega_white_alert),
                sdp_status=str(sdp_diag.status),
                lambda_positive_only=bool(lambda_positive_only),
                threshold_mode=str(threshold_mode),
                include_M0=bool(include_M0),
            )
            results.append(row)

            print(
                f"λ={lambda_true: .4f} | "
                f"N={n_samples:6d} | "
                f"power={power: .3f} "
                f"[{ci_low:.3f}, {ci_high:.3f}] | "
                f"λ_hat={row.lambda_fit_mean:.5f} ± {row.lambda_fit_std:.5f} | "
                f"A_dim={basis.A_dim} | "
                f"proj={basis.projection_norm:.3e} | "
                f"LR_thr={LR_threshold:.3f}"
            )

    if save_outputs:
        export_results(
            results=results,
            basis=basis,
            output_path=output_path,
            sdp_diagnostics=sdp_diag,
            config={
                "n_sim": int(n_sim),
                "n_null": int(n_null),
                "dim": int(dim),
                "n_noise": int(n_noise),
                "seed": int(seed),
                "seed_basis": int(seed_basis),
                "seed_sim": int(seed_sim),
                "sigma_obs": float(sigma_obs),
                "alpha_true": float(alpha_true),
                "alpha_level": float(alpha_level),
                "lambda_positive_only": bool(lambda_positive_only),
                "threshold_mode": str(threshold_mode),
                "include_M0": bool(include_M0),
                "subsystem_dim": None if subsystem_dim is None else int(subsystem_dim),
                "lambda_true_values": lambda_true_values,
                "n_samples_values": n_samples_values,
                "eps_num": float(eps_num),
                "sdp_check": bool(sdp_check),
                "white_alert_threshold": float(white_alert_threshold),
                "sdp_solver": sdp_solver,
                "witness_stability": bool(witness_stability),
                "n_witness_samples": int(n_witness_samples),
                "witness_perturb_scale": float(witness_perturb_scale),
                "witness_stability_cos_threshold": float(witness_stability_cos_threshold),
            },
            witness_diagnostics=witness_diag,
        )

    if make_plots and save_outputs:
        plot_power_curves(results, output_path)
        plot_power_heatmap(results, output_path)

    return results


# ============================================================
# Exports et figures
# ============================================================


def export_results(
    results: Sequence[ResultRow],
    basis: Basis,
    output_path: Path,
    config: Dict[str, object],
    sdp_diagnostics: Optional[RobustnessDiagnostics] = None,
    witness_diagnostics: Optional[Dict[str, float]] = None,
) -> None:
    """Exporte résultats, diagnostics et configuration."""
    if not results:
        raise ValueError("Aucun résultat à exporter.")

    rows = [asdict(row) for row in results]

    csv_file = output_path / "monte_carlo_power_results.csv"
    json_file = output_path / "monte_carlo_power_results.json"
    basis_file = output_path / "basis_diagnostics.json"
    config_file = output_path / "simulation_config.json"
    sdp_file = output_path / "sdp_psd_proxy_diagnostics.json"
    witness_file = output_path / "witness_stability_diagnostics.json"

    with csv_file.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    with json_file.open("w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2, ensure_ascii=False)

    with basis_file.open("w", encoding="utf-8") as f:
        json.dump(basis_diagnostics(basis), f, indent=2, ensure_ascii=False)

    with config_file.open("w", encoding="utf-8") as f:
        json.dump(config, f, indent=2, ensure_ascii=False)

    if sdp_diagnostics is not None:
        with sdp_file.open("w", encoding="utf-8") as f:
            json.dump(asdict(sdp_diagnostics), f, indent=2, ensure_ascii=False)

    if witness_diagnostics is not None:
        with witness_file.open("w", encoding="utf-8") as f:
            json.dump(witness_diagnostics, f, indent=2, ensure_ascii=False)

    print("\nExports :")
    print(f"- {csv_file}")
    print(f"- {json_file}")
    print(f"- {basis_file}")
    print(f"- {config_file}")
    if sdp_diagnostics is not None:
        print(f"- {sdp_file}")
    if witness_diagnostics is not None:
        print(f"- {witness_file}")


def plot_power_curves(results: Sequence[ResultRow], output_path: Path) -> None:
    """Courbes de puissance."""
    lambdas = sorted(set(row.lambda_true for row in results))
    plt.figure(figsize=(10, 6))
    for lam in lambdas:
        subset = sorted([row for row in results if row.lambda_true == lam], key=lambda row: row.n_samples)
        x = np.array([row.n_samples for row in subset], dtype=float)
        y = np.array([row.power for row in subset], dtype=float)
        y_low = np.array([row.power_ci_low for row in subset], dtype=float)
        y_high = np.array([row.power_ci_high for row in subset], dtype=float)
        plt.plot(x, y, marker="o", label=f"λ={lam:g}")
        plt.fill_between(x, y_low, y_high, alpha=0.12)
    plt.xscale("log")
    plt.ylim(-0.02, 1.02)
    plt.xlabel("Nombre d'échantillons N")
    plt.ylabel("Puissance statistique")
    plt.title("Puissance LR avec bruit calibré multidimensionnel")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    file_path = output_path / "power_curves.png"
    plt.savefig(file_path, dpi=180)
    plt.close()
    print(f"- {file_path}")


def plot_power_heatmap(results: Sequence[ResultRow], output_path: Path) -> None:
    """Carte de puissance."""
    lambdas = sorted(set(row.lambda_true for row in results))
    n_values = sorted(set(row.n_samples for row in results))
    grid = np.full((len(lambdas), len(n_values)), np.nan)
    index = {(row.lambda_true, row.n_samples): row.power for row in results}
    for i, lam in enumerate(lambdas):
        for j, n_samples in enumerate(n_values):
            grid[i, j] = index[(lam, n_samples)]
    plt.figure(figsize=(10, 6))
    image = plt.imshow(grid, aspect="auto", origin="lower", vmin=0.0, vmax=1.0)
    plt.colorbar(image, label="Puissance")
    plt.xticks(range(len(n_values)), [str(n) for n in n_values], rotation=45)
    plt.yticks(range(len(lambdas)), [f"{lam:g}" for lam in lambdas])
    plt.xlabel("Nombre d'échantillons N")
    plt.ylabel("λ vrai")
    plt.title("Carte de puissance Monte Carlo")
    plt.tight_layout()
    file_path = output_path / "power_heatmap.png"
    plt.savefig(file_path, dpi=180)
    plt.close()
    print(f"- {file_path}")


# ============================================================
# CLI
# ============================================================


def parse_float_list(text: str) -> List[float]:
    values = [float(x.strip()) for x in text.split(",") if x.strip()]
    if not values:
        raise argparse.ArgumentTypeError("La liste de valeurs lambda est vide.")
    return values


def parse_int_list(text: str) -> List[int]:
    values = [int(x.strip()) for x in text.split(",") if x.strip()]
    if not values:
        raise argparse.ArgumentTypeError("La liste de tailles d'échantillon est vide.")
    if any(value <= 0 for value in values):
        raise argparse.ArgumentTypeError("Toutes les tailles d'échantillon doivent être > 0.")
    return values


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Simulation Monte Carlo : N_cal multidimensionnel + espace admissible A."
    )
    parser.add_argument("--n-sim", type=int, default=5000, help="Nombre de simulations par point.")
    parser.add_argument("--n-null", type=int, default=20000, help="Simulations nulles en mode empirical.")
    parser.add_argument("--dim", type=int, default=16, help="Dimension matricielle du modèle toy.")
    parser.add_argument("--n-noise", type=int, default=3, help="Dimension du sous-espace de bruit calibré.")
    parser.add_argument("--seed", type=int, default=42, help="Graine aléatoire principale.")
    parser.add_argument(
        "--seed-basis",
        type=int,
        default=None,
        help="Graine dédiée à la construction de W0, N_cal, S_opt et A. Par défaut : --seed.",
    )
    parser.add_argument(
        "--seed-sim",
        type=int,
        default=None,
        help="Graine dédiée aux observations simulées et au bootstrap. Par défaut : --seed + 1000003.",
    )
    parser.add_argument("--sigma-obs", type=float, default=float(np.sqrt(0.0008)), help="Écart-type observationnel.")
    parser.add_argument("--alpha-true", type=float, default=0.12, help="Amplitude du bruit dominant dans W0.")
    parser.add_argument("--alpha-level", type=float, default=0.01, help="Seuil de significativité.")
    parser.add_argument(
        "--threshold-mode",
        choices=["asymptotic", "empirical"],
        default="asymptotic",
        help="Mode de calcul du seuil LR.",
    )
    parser.add_argument("--output-dir", type=str, default="monte_carlo_outputs_control", help="Dossier de sortie.")
    parser.add_argument(
        "--lambda-values",
        type=str,
        default="0,0.003,0.005,0.010",
        help="Liste des lambda vrais, séparés par des virgules. 0.005 est le seuil réaliste de preuve de concept ; 0.003 est aspirationnel.",
    )
    parser.add_argument(
        "--n-values",
        type=str,
        default="1000,2000,5000,10000,20000,50000,100000",
        help="Liste des tailles d'échantillon, séparées par des virgules.",
    )
    parser.add_argument(
        "--include-M0",
        action="store_true",
        help="Ajoute des contraintes toy de marges nulles via traces partielles.",
    )
    parser.add_argument(
        "--subsystem-dim",
        type=int,
        default=None,
        help="Dimension dA pour l'interprétation dim=dA*dB si --include-M0.",
    )
    parser.add_argument(
        "--eps-num",
        type=float,
        default=1e-8,
        help="Seuil numérique d'applicabilité sur ||P_A vec(S_opt)||. À pré-enregistrer.",
    )
    parser.add_argument(
        "--two-sided",
        action="store_true",
        help="Autorise lambda positif ou négatif. Par défaut, lambda >= 0.",
    )
    parser.add_argument(
        "--sdp-check",
        action="store_true",
        help="Exécute le contrôle SDP proxy régularisé par W_white (requiert cvxpy).",
    )
    parser.add_argument(
        "--white-alert-threshold",
        type=float,
        default=0.5,
        help="Seuil d'alerte sur omega_white.",
    )
    parser.add_argument(
        "--sdp-solver",
        type=str,
        default=None,
        help="Solveur cvxpy à utiliser pour --sdp-check, par ex. CLARABEL ou SCS.",
    )
    parser.add_argument(
        "--witness-stability",
        action="store_true",
        help="Exécute un diagnostic synthétique de stabilité de K_rel sous perturbations du witness.",
    )
    parser.add_argument(
        "--n-witness-samples",
        type=int,
        default=200,
        help="Nombre de perturbations pour le diagnostic --witness-stability.",
    )
    parser.add_argument(
        "--witness-perturb-scale",
        type=float,
        default=0.05,
        help="Amplitude des perturbations synthétiques du witness.",
    )
    parser.add_argument(
        "--witness-stability-cos-threshold",
        type=float,
        default=0.90,
        help="Seuil d'alerte sur le cosinus minimal entre K_rel et les directions perturbées.",
    )
    parser.add_argument("--no-plots", action="store_true", help="Désactive les graphiques.")
    parser.add_argument("--no-outputs", action="store_true", help="Désactive les exports.")

    args = parser.parse_args()

    run_monte_carlo_control(
        lambda_true_values=parse_float_list(args.lambda_values),
        n_samples_values=parse_int_list(args.n_values),
        n_sim=args.n_sim,
        n_null=args.n_null,
        dim=args.dim,
        n_noise=args.n_noise,
        seed=args.seed,
        seed_basis=args.seed_basis,
        seed_sim=args.seed_sim,
        sigma_obs=args.sigma_obs,
        alpha_true=args.alpha_true,
        alpha_level=args.alpha_level,
        lambda_positive_only=not args.two_sided,
        threshold_mode=args.threshold_mode,
        include_M0=bool(args.include_M0),
        subsystem_dim=args.subsystem_dim,
        eps_num=args.eps_num,
        output_dir=args.output_dir,
        save_outputs=not args.no_outputs,
        make_plots=not args.no_plots,
        sdp_check=bool(args.sdp_check),
        white_alert_threshold=float(args.white_alert_threshold),
        sdp_solver=args.sdp_solver,
        witness_stability=bool(args.witness_stability),
        n_witness_samples=int(args.n_witness_samples),
        witness_perturb_scale=float(args.witness_perturb_scale),
        witness_stability_cos_threshold=float(args.witness_stability_cos_threshold),
    )


if __name__ == "__main__":
    main()
