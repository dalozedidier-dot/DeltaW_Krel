# Convention officielle unique du dépôt ΔW/K_rel

Ce document fixe **la** convention utilisée partout dans le code, les tests et
les notebooks. Toute déviation est un bug. Référence externe :
Araújo, Branciard, Costa, Feix, Giarmatzi, Brukner, *Witnessing causal
nonseparability*, New J. Phys. **17**, 102001 (2015) — ci-après **ABCFGB**.

## 1. Ordre des espaces et dimensions

| Niveau | Ordre tensoriel | Dimensions | Constante code |
|---|---|---|---|
| Bipartite (sans futur) | `[AI, AO, BI, BO]` | qubits par défaut : (2, 2, 2, 2), D = 16 | `ProcessDims(2, 2, 2, 2)` |
| Avec futur global (switch) | `[AI, AO, BI, BO, F]` avec `F = F_target ⊗ F_control` (cible **puis** contrôle) | (2, 2, 2, 2, 4), D = 64 | `SWITCH_DIMS_WITH_FUTURE` |

Les indices de sous-systèmes sont `AI=0, AO=1, BI=2, BO=3` (et `F=4` avec
futur). Le futur `F` est un système **d'entrée uniquement** (pas de sortie).

## 2. Trace et normalisation

- Convention de trace : **Tr(W) = d_O = d_AO · d_BO** (le futur ne compte pas
  dans d_O). Pour tous les objets qubits du dépôt : Tr(W) = 4.
- Bruit blanc : `W_white = I_D · d_O / D` (bipartite : I/4 ; avec futur : I/16).
- Vérifiée par `assert_trace_convention` et par les tests.

## 3. Version complète vs version réduite du switch

- **Version complète (officielle)** : processus pur de rang 1 sur
  `[AI, AO, BI, BO, F]`, D = 64,

  `|w⟩ = (|w_{A≺B}⟩|0⟩_Fc + |w_{B≺A}⟩|1⟩_Fc)/√2`,
  `|w_{A≺B}⟩ = |ψ⟩_AI |1⟩⟩_{AO,BI} |1⟩⟩_{BO,Ft}`,
  `|w_{B≺A}⟩ = |ψ⟩_BI |1⟩⟩_{BO,AI} |1⟩⟩_{AO,Ft}`,

  avec `|1⟩⟩ = Σ_j |jj⟩`, contrôle initial `|+⟩`, cible initiale `|ψ⟩ = |0⟩`
  par défaut (amplitudes réelles ⇒ W réel symétrique). C'est la convention
  ABCFGB ; elle est **causalement non séparable** (R_g ≈ 0.5454).
- **Version réduite** : le switch **déphasé sur le contrôle**
  (`dephased_switch_process`) donne `(W_{A≺B≺F} + W_{B≺A≺F})/2`, causalement
  séparable (R_g = 0). Il sert de contrôle négatif du benchmark. La trace
  partielle sur F du switch redonne les objets bipartites de `switch_models`.

## 4. Caractérisation projective des sous-espaces

Tous les sous-espaces sont écrits avec la carte trace-and-replace
`τ_S W = (I_S/d_S) ⊗ Tr_S(W)` (ordre des facteurs préservé), selon la
caractérisation projective de Wechs–Abbott–Branciard (NJP 21, 013027, 2019).
Les conditions `[Π(1−τ_XO)] τ_(autres) W = 0` sont des projecteurs orthogonaux
deux à deux ; le projecteur sur l'intersection des noyaux est `1 − Σ Q_i`.

### Table d'audit : équation → fonction Python → test

| Équation (article/littérature) | Fonction | Tests |
|---|---|---|
| `τ_S W = (I_S/d_S) ⊗ Tr_S W` | `projectors.trace_replace` (explicite, auditable) ; `trace_replace_nd` (rapide, n systèmes) | `test_projectors_extended.py` (linéarité, idempotence, produits de Kronecker) ; égalité lente/rapide dans `test_quantum_switch.py` |
| `L_V` bipartite : `[1−B_O]A_I A_O W = 0`, `[1−A_O]B_I B_O W = 0`, `[1−A_O][1−B_O] W = 0` | `L_valid_bipartite` | `test_projectors.py`, `test_projectors_extended.py` (projection orthogonale, spectre {0,1}) |
| `L_{A≺B}` : `W = τ_BO W` et `[1−A_O]B_I B_O W = 0` ⇔ `L = τ_BO − τ_BIBO + τ_AOBIBO` | `L_A_before_B` | inclusion dans `L_V`, points fixes des processus d'ordre fixe (`test_switch_models_extended.py`) |
| `L_{B≺A}` (miroir) | `L_B_before_A` | idem |
| `L_{A≺B≺F}` : `(1−τ_BO)τ_F W = 0`, `(1−τ_AO)τ_BIBOF W = 0` | `L_A_before_B_with_future` | `test_quantum_switch.py` : idempotence, inclusion dans `L_valid_with_future`, réduction au cas sans futur pour d_F = 1, appartenance des branches du switch |
| `L_{B≺A≺F}` (miroir) | `L_B_before_A_with_future` | idem |
| `L_V^F` (validité avec futur) : `(1−τ_AO)τ_BIBOF W = 0`, `(1−τ_BO)τ_AIAOF W = 0`, `(1−τ_AO)(1−τ_BO)τ_F W = 0` | `L_valid_with_future` | `test_quantum_switch.py` : le switch est valide ; les combs sont inclus |
| `W_switch = |w⟩⟨w|` (ABCFGB) | `switch_models.ideal_quantum_switch_process` | rang 1, Tr = 4, PSD, symétrie d'échange, non-appartenance aux combs |
| `K_CS = C_{A≺B≺F} + C_{B≺A≺F}` ; robustesse généralisée `R_g = min Tr(X)/d_O`, `W + X = W_1 + W_2` | `sdp.solve_switch_generalized_robustness` | **benchmark** : `R_g(switch) ≈ 0.5454` (ABCFGB) ; `R_g(switch déphasé) = 0` |
| Robustesse au bruit blanc (random robustness) | `sdp.solve_switch_random_robustness` | valeur épinglée en régression (≈ 1.9908) |
| Robustesse calibrée bipartite | `sdp.solve_cs_robustness` | valeurs analytiques exactes (`test_sdp_extended.py`) |

## 5. Benchmark externe reproduit

- **Valeur publiée** : robustesse généralisée du quantum switch ≈ **0.5454**
  (ABCFGB, table des études de cas ; SDP dual = witness causal optimal).
- **Valeur calculée par le dépôt** : **0.545351** (SCS, eps = 1e-8, stable à
  eps = 1e-9), soit 0.5454 à 4 décimales ; écart ≤ 2·10⁻³ exigé par
  `run_sdp_validation.py` (`benchmark_passed`).
- Certificat dual : le witness S (dual de la contrainte d'égalité) satisfait
  `Tr(S·W_switch) = R_g` à ~10⁻⁹ près (champ `witness_certificate_gap`).
- Solveur : **SCS par défaut pour le benchmark switch** — CLARABEL échoue
  actuellement en `NumericalError` sur cette instance (D = 64) ; la
  configuration préenregistrée garde CLARABEL comme solveur primaire pour les
  cibles bipartites et documente ce fallback. Contre-vérification MOSEK
  optionnelle si licence disponible.

## 6. Statut des simulations (à ne pas surinterpréter)

- `scripts/monte_carlo_control_supplement.py` : **banc géométrique linéarisé**
  (contrôle méthodologique de la chaîne statistique ΔW/K_rel). Ce n'est ni une
  simulation physique du switch ni une validation expérimentale.
- `scripts/micro_tomography_simulation.py` : **stress test simplifié / preuve
  de concept** (modèle binomial blanchi à un paramètre), pont vers une future
  chaîne SDP + MLE complète.
- Seul le niveau SDP (`run_sdp_validation.py`) porte le benchmark externe
  (switch idéal). Les conclusions du manuscrit doivent citer chaque niveau
  avec son statut exact.
