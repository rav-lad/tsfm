# TSFM — Référence technique : données synthétiques

**Projet :** Time Series Foundation Model (< 30 millions de paramètres)  
**Périmètre :** conception, génération, validation, stockage et échantillonnage des données  
**Version :** 1.0 — 7 novembre 2026  
**Statut :** spécification initiale à valider expérimentalement  
**Principe non négociable :** aucun exemple réel n'entre dans le préentraînement du modèle.

> **Question de recherche.** Jusqu'où peut généraliser un TSFM de moins de 30M de paramètres, entraîné exclusivement sur une distribution **compositionnelle** de processus synthétiques ?
>
> **Idée directrice.** Nous n'essayons pas d'imiter un dataset réel précis. Nous échantillonnons des *mécanismes génératifs* et entraînons un modèle à prédire leurs suites à partir d'historiques partiels.

---

## Table des matières

1. [Décisions et principes](#1-décisions-et-principes)
2. [Fondements scientifiques](#2-fondements-scientifiques)
3. [Qu'est-ce qu'une donnée d'entraînement ?](#3-quest-ce-quune-donnée-dentraînement-)
4. [Catalogue des générateurs temporels](#4-catalogue-des-générateurs-temporels)
5. [Génération multivariée et covariables](#5-génération-multivariée-et-covariables)
6. [Distributions des paramètres et mélange des générateurs](#6-distributions-des-paramètres-et-mélange-des-générateurs)
7. [Volumes nécessaires et budget de calcul](#7-volumes-nécessaires-et-budget-de-calcul)
8. [Longueurs, horizons et échantillonnage](#8-longueurs-horizons-et-échantillonnage)
9. [Augmentations, valeurs manquantes et normalisation](#9-augmentations-valeurs-manquantes-et-normalisation)
10. [Pipeline, API et stockage](#10-pipeline-api-et-stockage)
11. [Train / validation / test et prévention des fuites](#11-train--validation--test-et-prévention-des-fuites)
12. [Contrôle qualité et diagnostics](#12-contrôle-qualité-et-diagnostics)
13. [Comment savoir si les données sont suffisantes ?](#13-comment-savoir-si-les-données-sont-suffisantes-)
14. [Expériences d'ablation](#14-expériences-dablation)
15. [Recette de démarrage et configuration YAML](#15-recette-de-démarrage-et-configuration-yaml)
16. [Plan d'implémentation et critères d'acceptation](#16-plan-dimplémentation-et-critères-dacceptation)
17. [FAQ / décisions justifiées](#17-faq--décisions-justifiées)
18. [Bibliographie commentée](#18-bibliographie-commentée)

---

## 1. Décisions et principes

### 1.1. Ce qu'on veut produire

Un **générateur d'épisodes de forecasting** à partir de séries temporelles entièrement synthétiques, plutôt qu'un CSV géant contenant uniquement des courbes préfabriquées.

Un *épisode* spécifie :

- Un historique observé de `L` points et un horizon de prédiction de `H` points.
- Une ou plusieurs **cibles** à prédire.
- Éventuellement des **covariables passées** et des **covariables futures connues**.
- Des masques précisant les valeurs visibles et celles utilisées pour la loss.
- Les mécanismes synthétiques ayant produit l'épisode (pour l'audit et les ablations ; **jamais fournis au modèle**).

### 1.2. Décisions V1

| Question | Décision initiale | Niveau de confiance |
|---|---|---|
| Source des données de training | 100 % synthétique | Contrainte du projet |
| Univarié ou multivarié ? | Univarié d'abord, multivarié ensuite | Choix pragmatique |
| Cible | Forecasting, puis quantiles probabilistes | Appuyé par Timestral / t0 |
| Génération | **Hybride** : à la volée pour le simple, cache pour le coûteux | Choix d'ingénierie |
| Taille du corpus | Pilotage par **épisodes vus** et **positions de loss**, pas nombre de fichiers | Choix méthodologique |
| Générateurs principaux | 7 familles temporelles + opérateurs relationnels | Appuyé par Chronos / TempoPFN / Timer-M1 / t0 |
| Longueurs/horizons | Mélange de valeurs, non fixes | Appuyé par TempoPFN / Timestral |
| Normalisation | Calculée sur **le passé effectivement visible** | Anti-leakage ; Timer-M1 |
| Vérification OOD | Holdouts de paramètres, familles et graphes | Recommandation de recherche |
| Benchmark réel | **Évaluation uniquement**, après gel du protocole si « synth-only » strict | Anti-contamination |

**Important :** les pourcentages, plages de paramètres, volumes et seuils écrits ci-dessous sont **nos hypothèses de départ**. Les papers justifient des *familles de mécanismes* et des *principes*, pas automatiquement nos hyperparamètres précis.

### 1.3. Ce que « 100 % synthétique » veut dire exactement

- **Autorisé :** utiliser des articles pour sélectionner des générateurs ; échantillonner leurs paramètres ; créer des séries à partir de RNG, d'équations, de graphes et de kernels ; évaluer sur des séries réelles non utilisées pour entraîner.
- **Interdit dans le training :** préentraîner sur GIFT-Eval, ETTh, ETTm, Weather, M4, M5, etc., puis affirmer « entraîné exclusivement sur données synthétiques » ; fabriquer du synthétique à partir de **fenêtres réelles copiées ou mélangées**.
- **À déclarer :** régler les distributions synthétiques après avoir examiné les scores d'un benchmark réel constitue une forme d'**adaptation indirecte au benchmark**. Cela ne modifie pas les poids à partir de données réelles, mais affaiblit la revendication de généralisation strictement zéro-shot. Idéalement : gel des hyperparamètres sur validation synthétique, puis une évaluation réelle finale.

---

## 2. Fondements scientifiques

### 2.1. L'idée d'un prior génératif

On choisit une famille de générateurs `g`, des paramètres `θ`, des bruits aléatoires `ε`, puis on fabrique :

\[
 g \sim p(g),\qquad \theta\sim p(\theta\mid g),\qquad X=G_g(\theta,\varepsilon).
\]

Un épisode sélectionne ensuite les variables observées, la coupure temporelle et l'horizon :

\[
 E = \operatorname{Episode}(X, L,H,\text{roles},\text{masks}).
\]

Le préentraînement apprend à approximer :

\[
 p(Y_{T+1:T+H}\mid Y_{1:T},X^{past},X^{known\ future}).
\]

Avec une loss de quantiles, on approche des quantiles **marginaux** de cette distribution conditionnelle ; cela ne garantit pas, à lui seul, des trajectoires futures conjointement cohérentes.

**Justification :** les approches *Prior-Data Fitted Networks* (ForecastPFN, TempoPFN) associent la généralisation au prior défini par les données générées. [R1, R2]

### 2.2. Ce que les modèles récents démontrent — et ne démontrent pas

| Publication | Fait vérifié pertinent | Ce que nous en empruntons | Limite de l'analogie |
|---|---|---|---|
| **ForecastPFN** (2023) | Modèle zero-shot entraîné sur une distribution synthétique | Générateurs compositionnels de tendances/saisonnalités | Domaine et architecture spécifiques |
| **Chronos** (2024) | `KernelSynth` compose des kernels de processus gaussiens | Banque de kernels + additions / produits | Chronos utilise aussi des **données réelles** : sa performance ne prouve pas qu'un mélange 100 % GP suffit |
| **TempoPFN**, v5 (2026) | Corpus d'~**10M séries synthétiques**, générateurs variés ; ablation SDE marquée | SDE, discontinuités, phénomènes asymétriques, pipeline reproductible | Matériel et budget d'entraînement beaucoup plus importants |
| **Timer-M1** (2026) | *Temporal primitives* + *relational primitives* + épisodes de rôles | Dépendances, graphes, sélection des covariables | Sa recette complète mélange réel et synthétique |
| **t0** — rapport t0-alpha/beta (2026) | Kernels, graphes causaux, effets de covariables, *decoy covariates*, curriculum | Covariables utiles/inutiles, disponibilité future, curriculum | **Le rapport ne décrit pas une recette synth-only de t0-gamma** ; ne pas attribuer ses chiffres d'entraînement à gamma |
| **Timestral** (2026) | **Préentraînement** synth-only, puis **post-training** réel + synthétique ; séries uni-/multivariées | Diversité, rôles, bruit, régimes, comptages | Le checkpoint publié **n'est pas entraîné exclusivement sur du synthétique de bout en bout** |

Faits particulièrement utiles :

- TempoPFN rapporte environ **10M séries** (de 500k à 2M par générateur) et une dégradation relative d'environ **26 % de CRPS** quand le générateur SDE est retiré *dans leur configuration d'ablation* [R2]. Ce n'est **pas** une preuve que 10M sont nécessaires, ni que notre mélange optimal doit reproduire le leur.
- Dans l'ablation de données de Timer-M1, la recette synth-only obtient un FEV skill score de **0,4710** contre **0,4887** pour la recette complète. Le synthétique seul est compétitif, mais pas identique à l'utilisation conjointe des données réelles [R3].
- t0-alpha augmente progressivement le poids du pool multivarié (**10 % → 20 % → 40 %**) : c'est un précédent crédible pour un *curriculum*, pas une obligation de copier ces proportions [R4].
- Timestral couvre pendant son préentraînement synthétique tendances, saisonnalités, bruit, ruptures de régime, événements localisés, changements de niveau, comptages et relations inter-séries [R5].

### 2.3. Notre hypothèse principale

La diversité structurelle des générateurs compte autant, voire davantage, que le simple accroissement du nombre de séries. Nous testerons cette hypothèse en **fixant le budget d'entraînement** et en faisant varier soit les volumes, soit les familles de générateurs.

---

## 3. Qu'est-ce qu'une donnée d'entraînement ?

**Ne pas confondre :**

1. **Source générative** : réalisation longue d'un mécanisme `G(θ, ε)`.
2. **Record** : tableau multivarié aligné `values[time, channels]`, avec son graphe et ses méta-informations.
3. **Épisode** : exemple que voit le réseau, avec origine de prévision, `L`, `H`, rôles, masques.
4. **Position supervisée** : valeur cible au futur réellement incluse dans la loss.

Une même source peut produire plusieurs épisodes, **mais** ces épisodes sont corrélés. Dix fenêtres extraites d'une même sinusoïde ne valent pas dix processus indépendants.

### 3.1. Contrat d'un épisode (schéma canonique)

```python
{
    "values": float32[C, L + H],  # valeurs synthétiques complètes, côté dataset
    "observed_mask": bool[C, L + H],
    "loss_mask": bool[C, H],       # seuls les futurs des cibles valides sont scorés
    "roles": int[C],              # TARGET, PAST_ONLY, KNOWN_FUTURE
    "context_length": int,
    "prediction_length": int,
    "origin": int,
    "meta": {
        "generator_family": str,
        "generator_version": str,
        "seed": int,
        "latent_parameters": dict,
        "graph": dict | None,
        "augmentations": list,
        "split": str,
    }
}
```

**ATTENTION :** `values` contient les labels futurs **uniquement dans la mémoire du dataset/trainer**. Le modèle reçoit une vue masquée `model_input = where(observed_mask, values, 0)`, jamais le tenseur brut contenant le futur cible.

Pour une *past-only covariate*, son futur existe dans les données générées, mais **est systématiquement masqué à l'entrée**. Pour une *known-future covariate*, les valeurs futures sont visibles **uniquement si elles seraient réellement connues à l'origine du forecast**.

### 3.2. Démarrage univarié

- `C = 1`, une cible.
- `observed_mask[:L] = 1`, `observed_mask[L:] = 0`.
- `loss_mask = 1` pour les horizons valides.
- En V2, ajouter des covariables avec leurs propres masques d'observation.

---

## 4. Catalogue des générateurs temporels

Les sept familles ci-dessous forment notre **prior initial**. Elles peuvent être échantillonnées seules ou combinées.

### G1 — Tendances + saisonnalités classiques

**Mécanisme :**

\[
 y_t=b_0+b_1t+b_2t^2+\sum_{k=1}^{K}A_k\sin(2\pi t/P_k+\phi_k)+\epsilon_t.
\]

Extensions : croissance exponentielle **bornée**, tendance locale par morceaux, saisonnalité multiplicative `level × seasonality`, amortissement de l'amplitude.

**Paramètres de départ :** `K ∈ {0,1,2,3}`, amplitude par rapport à l'échelle du signal, périodes liées à la longueur observée, niveaux de bruit faibles/modérés/forts.

**Pourquoi :** ForecastPFN ; base identifiable de tendances et périodicités [R1].

**Tests QA :** vérification de la fréquence dominante ; extrapolation de pente ; éviter que les tendances exponentielles débordent.

### G2 — Périodicité complexe et asymétrie

Sawtooth, square wave, chirp, cycles asymétriques, impulsions saisonnières, périodes multiples, amplitude ou période variant lentement.

**Pourquoi :** TempoPFN souligne explicitement les motifs asymétriques et les discontinuités [R2].

**QA :** anti-aliasing lors du rééchantillonnage ; vérifier les cycles suffisamment observés ; distinguer signaux périodiques propres et événements impulsionnels.

### G3 — Gaussian Processes / KernelSynth

\[
 f\sim\mathcal{GP}(m,k),\quad k=k_1+k_2\quad\text{ou}\quad k=k_1k_2.
\]

Banque initiale : `RBF`, `Matérn`, `Periodic`, `Linear`, `RationalQuadratic`, `WhiteNoise`. Échantillonner 1 à 4 kernels, une somme ou un produit valide ; vérifier la PSD numérique de la covariance.

**Pourquoi :** KernelSynth de Chronos [R6], variantes de TempoPFN [R2].

**Coût :** l'échantillonnage GP exact peut nécessiter une factorisation de coût en `O(T^3)` ; limiter la longueur GP exacte (p. ex. `T <= 512` au pilote), puis tester les Random Fourier Features, des méthodes structurées ou un cache [R2, R7].

**QA :** espacement des points, jitter diagonal pour la stabilité, corrélation empirique et smoothness, diversité des longueurs d'échelle. Attention : certains produits de kernels donnent des signaux presque plats ou du bruit ; les garder dans une certaine mesure, pas exclusivement.

### G4 — AR / ARMA / random walk / mean reversion

\[
 y_t=\sum_{i=1}^{p}\phi_i y_{t-i}+\epsilon_t+\sum_{j=1}^{q}\theta_j\epsilon_{t-j}.
\]

Inclure : AR(1), AR(p) **stable**, ARMA, random walk, moyenne régressive et **bruit blanc pur**.

**Pourquoi :** dynamique temporelle élémentaire ; besoin de cas où prédire la dernière valeur ou la moyenne est optimal ou presque.

**QA :** vérifier les racines du polynôme AR avant de considérer le processus stationnaire ; garder volontairement des processus non stationnaires (random walks) identifiés comme tels.

### G5 — SDE / changements de régime / volatilité

\[
 dy_t=\theta_{r_t}(\mu_{r_t}-y_t)\,dt+\sigma_{r_t}(t)\,dW_t,
\]

avec `r_t` chaîne de Markov, éventuels changements de tendance, niveau ou variance ; simulation par Euler–Maruyama avec pas de temps fixé et documenté.

**Pourquoi :** composant à fort impact dans l'ablation TempoPFN [R2].

**QA :** fréquence effective des transitions, fraction de périodes calmes/agitées, dispersion et autocorrélation par régime ; absence d'explosions numériques ; ne pas rendre chaque changement de régime parfaitement prévisible.

### G6 — Événements, spikes, intermittence et comptages

Exemples : demandes intermittentes (`Poisson`, `negative-binomial`, processus à zéro inflation), événements groupés, impulsions, anomalies isolées, plateaux, ruptures de niveau, promotions simulées.

**Pourquoi :** TempoPFN inclut anomalies, steps, spikes et rhythms ; Timestral mentionne événements localisés et observations de comptage [R2, R5].

**QA :** proportion de zéros, variance/moyenne des comptages, taille des pics, distribution des intervalles entre événements, contrôle des valeurs négatives pour les comptes.

### G7 — Compositions non linéaires

\[
 y_t=T(t)+A(t)S(t)+R(t)+\epsilon_t,
\]

avec transformations telles que `tanh`, saturation, clipping **ponctuel**, transformations d'échelle, dépendances entre tendance et bruit, bruit hétéroscédastique.

**Pourquoi :** en dehors des processus simples, l'assemblage crée des combinaisons qui forcent la généralisation ; Chronos, TempoPFN, Timer-M1 [R2, R3, R6].

**QA :** vérifier que la composition ne dégénère pas systématiquement en bruit pur ; pas de clipping excessif ; mesurer la diversité après transformation.

### 4.1. Couverture structurelle minimale

Notre générateur doit représenter les axes suivants indépendamment du nom de la famille :

| Axe | Cas nécessaires | Diagnostic |
|---|---|---|
| Prédictibilité | facile, intermédiaire, quasi impossible | skill vs naive/oracle |
| Stationnarité | stationnaire, tendance, random walk | ADF/KPSS indicatifs, tendances |
| Régularité | lisse, rugueux | variation totale, énergie haute fréquence |
| Saisonnalité | absente, simple, multiple, mouvante | spectre/ACF |
| Discontinuités | aucun, step, spike, burst | taux de ruptures/pics |
| Variance | constante, hétéroscédastique | fenêtres de variance |
| Support | continu signé, positif, entier | histogrammes/support |
| Observabilité | complet, gaps, blocs absents | taux/longueur de gaps |

Un contrôle simple : générer **au moins 100 exemples par famille** et inspecter des grilles avec des axes cohérents. Les images ne remplacent pas les tests statistiques.

---

## 5. Génération multivariée et covariables

### 5.1. Deux niveaux : primitives temporelles et primitives relationnelles

Nous générons d'abord `K` sources temporelles indépendantes `z_1, ..., z_K` (familles G1–G7). Puis nous construisons des réponses `x_i` à partir de ces sources via des opérateurs relationnels.

**Lien scientifique :** c'est l'organisation *temporal/relational primitives* de Timer-M1 et les familles de covariables de t0 [R3, R4].

### 5.2. Catalogue d'opérateurs relationnels

| Code | Opérateur | Exemple |
|---|---|---|
| R1 | Combinaison signée | `y_t = 0.7 z1_t - 0.3 z2_t + ε_t` |
| R2 | Décalage temporel | `y_t = a z1_(t-τ) + ε_t`, `τ >= 0` |
| R3 | Non-linéarité | `y_t = tanh(a z1_t + b z2_t) + ε_t` |
| R4 | Facteur commun | `x1_t = z_t + η1_t`, `x2_t = b z_t + η2_t` |
| R5 | Changement de réponse selon régime | `y_t = a_(r_t) z_t + ε_t` |
| R6 | Effet d'événement connu | `y_t = baseline_t + a * event_t + ε_t` |
| R7 | Cointegration synthétique | `x1_t = z_t + u_t`, `x2_t = β z_t + v_t` (stationnaires `u, v`) |
| R8 | Decoy / bruit non informatif | `x_decoy_t` indépendant de la cible |

**Graphe :** produire un DAG temporel (ou un SCM acyclique dans l'ordre de calcul) ; échantillonner les nœuds parents et leurs paramètres. Éviter des cycles instantanés implicites ; si des feedbacks sont désirés, utiliser des **lags strictement positifs**.

### 5.3. Rôles : target, past-only, known-future

1. **TARGET** : historique visible, futur caché et utilisé comme label.
2. **PAST_ONLY** : historique visible, futur caché, jamais supervisé dans l'épisode.
3. **KNOWN_FUTURE** : historique + futur visibles, parce que la variable serait *connue au cutoff* (calendrier, planning, promotion planifiée, calendrier d'événements simulé).

**Règle centrale :** une variable synthétique qui **cause** la cible future n'est **pas automatiquement** connue à l'avance. Son futur ne peut être visible que si son mécanisme correspond à une information ex ante disponible. Une « météo future parfaitement connue », utilisée sans incertitude comme une vraie prévision météo, crée une tâche irréaliste : en V1, réserver le rôle known-future au calendrier/planning ; en V2, générer séparément une **prévision bruitée** d'un facteur futur.

**Exemple correct :** `promo[t]` est fixé à `T` pour `t > T`; `sales[t]` dépend de `promo[t]` et d'un choc inconnu ; la promotion future est visible, le choc et les ventes futures ne le sont pas.

### 5.4. Contrôles de pertinence causale

Inclure dans chaque batch multivarié **des relations utiles, faibles et nulles** : un modèle ne doit pas supposer que toute covariable est informative.

Tests automatiques :

- **Covariate lift** : l'oracle simple `f(y_history, x)` doit mieux prévoir certaines tâches avec covariables qu'en les retirant.
- **Decoy test** : ajouter des covariables indépendantes ne doit pas améliorer systématiquement la prédiction.
- **Permutation test** : décaler ou mélanger un driver connu doit dégrader les cas où il est nécessaire.
- **No leakage test** : modifier arbitrairement les futurs masqués (`target_future`, `past_only_future`) **sans modifier les entrées visibles** doit laisser les entrées du modèle, donc sa prédiction, identiques.

### 5.5. Combien de variables ?

**Proposition, non tirée telle quelle des papers :**

- Prototype : `C = 1`.
- Premier multivarié : `C in {2, 3, 4}`.
- Extension : `C in {2, 4, 6, 8}` ; échantillonner le nombre de cibles, de covariables passées et futures.
- Exemples d'ablation : `C = 1` vs `C = 2–4` vs `C = 2–8` à budget égal.

Ne jamais comparer le volume d'un corpus multivarié et univarié sur le **nombre de fenêtres uniquement** : le nombre de valeurs et le coût attentionnel sont différents.

---

## 6. Distributions des paramètres et mélange des générateurs

### 6.1. Première distribution proposée (univariée)

| Famille | Poids initial | Justification |
|---|---:|---|
| G1 tendances/saisonnalités | 15 % | motifs universels |
| G2 périodicité complexe | 10 % | asymétrie, harmoniques |
| G3 GP | 15 % | prior non paramétrique |
| G4 AR/stochastique | 15 % | dépendance et limites de prédictibilité |
| G5 régimes/SDE | **20 %** | importance constatée dans TempoPFN |
| G6 événements/comptages | 10 % | ruptures, intermittence |
| G7 composites | 15 % | généralisation par recomposition |
| **Total** | **100 %** | **à tester** |

**Pourquoi ne pas équilibrer 1/7 ?** Un équilibre strict par nom de générateur ne garantit pas un équilibre en propriétés statistiques. Le surpoids de G5 est un **pari initial**, pas une conclusion directe de l'ablation de TempoPFN.

### 6.2. Paramètres : privilégier les ratios sans dimension

Ne pas utiliser des plages absolues uniformes sans relation à `L`, `H` ou à l'échelle de la série. Exemples de prior initial :

| Variable | Distribution proposée | Raison |
|---|---|---|
| Nombre de saisonnalités | `K ∈ {0,1,2,3}` | cas simples et composés |
| Période saisonnière | mélange de `P ∈ {7,24,48,96,168,365}` + `P/L` échantillonné | courte/longue ; dépend de la fenêtre |
| Amplitude | `sign × LogUniform(0.1, 3)` (si signé) | couvrir plusieurs ordres d'échelle |
| Bruit / signal (SNR) | trois régimes faible/moyen/fort | niveaux de difficulté |
| AR(1) stable | `φ ∈ [-0.95,0.95]`, option proche de ±1 | dépendance courte/longue |
| Lags relationnels | `τ ∈ {0,...,min(32,L/4)}` | éviter des effets systématiquement invisibles |
| Transitions de régime | `p_stay ∈ [0.90,0.999]` | régimes de durée variable |
| Nb de parents dans DAG | `0–3` | sparsité contrôlable |
| Fraction de decoys | `0–50 %` des covariables d'un épisode | robustesse à l'information inutile |
| Dégradation de l'observation | 0, puis scénarios jusqu'à 30 % | curriculum de missingness |

**Tous ces nombres sont des *defaults à ablater*.** Certaines valeurs de périodicité n'ont de sens que si suffisamment de cycles sont visibles ; un effet de période 365 dans un contexte 128 sera souvent impossible à identifier. On veut quelques exemples intrinsèquement ambigus pour apprendre l'incertitude, mais pas un corpus dominé par eux.

### 6.3. Ne pas rendre tous les exemples « faciles »

Un TSFM probabiliste doit apprendre des séries :

- Prédictibles (saisonnalité stable, AR peu bruité).
- Partiellement prédictibles (drivers observés, bruit résiduel).
- Quasi imprévisibles (bruit blanc, chocs futurs indépendants).

Sans le troisième groupe, le modèle peut devenir systématiquement **trop confiant** ; avec trop de bruit pur, il peut apprendre à prévoir la moyenne et ignorer les structures. Timestral décrit le prior comme déterminant aussi l'incertitude de la prédiction [R5].

### 6.4. Composition hiérarchique recommandée

```text
Famille dominante (G1..G7)
    ├── Paramètres structuraux
    ├── Bruit et dynamique résiduelle
    ├── Composante secondaire éventuelle
    ├── Transformations compatibles avec le support
    └── Record / événement de forecasting
```

Conserver en métadonnées la **famille dominante** et la liste des composants secondaires. Sans cela, il devient difficile d'attribuer les résultats aux bonnes familles dans les ablations.

---

## 7. Volumes nécessaires et budget de calcul

### 7.1. Réponse : « combien de données nous faut-il ? »

**Il n'existe pas de nombre universel démontré pour un TSFM < 30M entraîné uniquement sur du synthétique.** Le nombre optimal dépend du modèle, du contexte, de l'horizon, de la diversité et du nombre de tokens/positions effectivement supervisées.

**Ordres de grandeur utiles pour notre projet :**

| Niveau | Épisodes exposés au modèle* | Objectif |
|---|---:|---|
| Smoke test | **1k–10k** | formes, masks, gradients, débit |
| Prototype | **20k–100k** | modèle capable d'apprendre des processus simples |
| Pilot 5M | **200k–500k** | premiers gains sur validation synthétique OOD |
| Préentraînement exploratoire | **1–3M** | mesurer scaling et effet du mélange |
| Préentraînement ambitieux | **3–10M** | cible possible si compute disponible et gains observés |

\* **Épisodes exposés ≠ épisodes uniques.** En génération en ligne, un même mécanisme peut être rééchantillonné, et plusieurs épisodes peuvent provenir d'une même source. Il faut enregistrer les **exposures réelles**, les **sources indépendantes** et les **positions de loss**. Ce sont des paliers de planification, pas des recommandations issues d'une équation universelle.

**Point de comparaison :** TempoPFN utilise ~10M séries synthétiques dans son corpus et décrit un entraînement substantiellement plus lourd que ce projet [R2]. Nous ne devons pas copier son volume sans mesurer notre débit.

### 7.2. Trois métriques de quantité à tracer

1. `n_source_draws` : nombres de processus/source graphs indépendamment échantillonnés.
2. `n_episodes_seen` : nombre d'épisodes présentés au réseau (inclut répétitions/augmentations).
3. `n_supervised_scalar_positions` : somme des positions effectivement scorées :

\[
N_{loss}=\sum_{e=1}^{N_{episodes}}C_{target,e}\,H_e\times\text{fraction de futurs valides}.
\]

Tracer également `n_input_scalar_positions`, `n_input_tokens` (après patching) et la proportion de chaque famille réellement vue.

### 7.3. Exemples de budget (calculs transparents)

Pour `L=512`, `H=96`, `C=1`, valeurs en `float32`, **sans** masques ni métadonnées :

- `1M épisodes × 608 pas × 4 octets ≈ 2,43 GB` bruts.
- `10M épisodes ≈ 24,32 GB` bruts.
- Avec `C=4` : **~9,73 GB** pour 1M épisodes.

Les données effectivement stockées peuvent être plus volumineuses ; la mémoire d'entraînement dépend aussi des activations du modèle et des masques.

Exemple côté optimisation : `20 000 steps × batch size 64 = 1,28M épisodes vus`. Ce nombre **ne garantit pas** 1,28M sources indépendantes. Avec `H=96` et une cible complète, cela donne `122,88M positions supervisées`.

### 7.4. Comment choisir la quantité finale ?

**Plan :** entraîner une même architecture avec trois budgets, par exemple `0,2M`, `1M`, `3M` épisodes vus, en conservant autant que possible même distribution, même protocole et même compute de référence. Mesurer :

- Score sur validation synthétique ID.
- Score sur validation synthétique OOD (paramètres/familles/relations).
- Gain marginal par heure GPU.
- Évolution du calibrage et du *covariate lift*.

Si tripler les exemples n'améliore presque pas la validation OOD, essayer **une nouvelle famille / un meilleur mélange**, plutôt que multiplier aveuglément les séries.

### 7.5. Génération à la volée ou dataset pré-calculé ?

**Choix : hybride.**

| Type | Stratégie proposée | Motif |
|---|---|---|
| Sinusoïdes / tendances / AR / spikes / counts | En ligne, sur CPU | rapide, diversité illimitée |
| SDE | En ligne si débit suffisant ; sinon pool pré-calculé | coût modéré/variable |
| GP exact / SCM avec GP | Cache de sources + épisodes dynamiques | coûteux, génération parallèle |
| Validation et tests synthétiques | Pré-calculés **et gelés** | comparabilité stricte |

**Règle :** le cache accélère l'entraînement mais ne doit pas réduire la diversité à quelques centaines de courbes vues des milliers de fois. Mesurer le taux de réutilisation et les seeds distinctes.

### 7.6. Ce qui fera réellement varier le budget

- Le coût de génération (`series/sec`, par famille et par longueur).
- Les longueurs `L+H`.
- Le nombre de canaux `C`.
- Les patches/positions d'attention.
- Le nombre de modèles d'ablation et de seeds.

**Avant de réserver du GPU, profiler** `temps génération`, `temps forward+backward`, `% temps GPU en attente DataLoader` et `GB/s disque`. On adaptera la quantité aux premiers profils mesurés.

---

## 8. Longueurs, horizons et échantillonnage

### 8.1. Pourquoi échantillonner des longueurs différentes ?

Le modèle doit fonctionner sur des histoires courtes ou longues et prédire plusieurs horizons. Un seul couple `(L=512,H=96)` encouragerait une spécialisation inutile.

**Proposition de pilote :**

| Longueur historique `L` | Probabilité |
|---:|---:|
| 128 | 15 % |
| 256 | 25 % |
| 512 | 40 % |
| 1024 | 20 % |

| Horizon `H` | Probabilité |
|---:|---:|
| 24 | 20 % |
| 48 | 25 % |
| 96 | 35 % |
| 192 | 20 % |

Ces distributions sont **nos defaults**, non la recette de Timestral. TempoPFN échantillonne lui aussi différentes longueurs, avec une forte préférence pour 2048 dans sa recette complète [R2].

Échantillonner `H` sous contrainte de la longueur de la source. Garder une petite fraction de cas `H/L` élevé pour tester les extrapolations difficiles.

### 8.2. Fréquences et saisonnalités

Le modèle ne recevra pas nécessairement un timestamp réel. Nous pouvons simuler :

- Des périodes définies en **nombre de pas** (`24`, `7`, `168`, etc.).
- Des cycles à temps normalisé, afin de diversifier les échelles.
- Des calendriers explicites dans les scénarios avec **known-future covariates**.

Ne pas confondre « période de 24 pas » avec « cycle journalier » : l'interprétation dépend de la fréquence de sampling. En absence de métadonnée de fréquence, le modèle apprend des motifs en pas de temps, pas une sémantique calendrier universelle.

### 8.3. Échantillonnage des fenêtres

Pour une source de longueur `T_source >= L+H+buffer`, choisir une origine `t0` telle que :

```text
history = X[t0-L : t0]
future  = X[t0 : t0+H]
```

Les lags causaux nécessitent un `buffer` initial. On peut utiliser plusieurs origines sur une même source **en training**, mais pas répartir des fenêtres d'une même source entre train et validation/test.

### 8.4. Une remarque importante sur les pertes

Quand les horizons sont mélangés, la loss doit être normalisée par le **nombre de positions scorées** et non par la longueur paddée. Sinon, les épisodes à horizon long ou avec davantage de cibles peuvent dominer involontairement.

---

## 9. Augmentations, valeurs manquantes et normalisation

### 9.1. Augmentations recommandées

| Augmentation | Utilisation | Risque à contrôler |
|---|---|---|
| Scale positive aléatoire | invariance d'unité | overflow / séries de comptages |
| Offset aléatoire | invariance du niveau | support non négatif |
| Sign inversion | symétrie quand pertinent | ne pas l'appliquer aux demandes/comptages positifs |
| Mélange de composantes synthétiques | diversifier motifs | moyennage excessif, signal trop lisse |
| Noise injection | robustesse | détruire l'information prédictive |
| Resampling / time warping | rythmes différents | aliasing ; casser alignement inter-séries |
| Missingness ponctuelle | robustesse | confusion avec valeurs zéro |
| Blocs manquants | pannes/capteurs | trous non représentatifs |
| Outliers / spikes | anomalies | robustesse excessive au détriment des vrais pics |

**Ne pas appliquer chaque augmentation à chaque épisode.** Échantillonner `none / one / multiple` et sauvegarder les opérations dans `metadata`.

Pour le multivarié, toute transformation temporelle qui doit préserver les relations **s'applique conjointement aux variables alignées**. Une permutation temporelle indépendante des canaux peut être utile comme **contre-exemple**, mais pas comme exemple « relation conservée ».

### 9.2. Masques et missingness

Distinguer quatre choses :

1. **Zéro observé**, vrai signal `0`.
2. **Valeur absente**, `observed_mask=0`.
3. **Futur cible masqué**, `observed_mask=0`, `loss_mask=1`.
4. **Padding**, ni observation ni supervision.

Le modèle doit recevoir **valeurs masquées + indicateur de visibilité** (voire un code de type de masque). Timer-M1 inclut explicitement le masque d'observation dans l'embedding des patches [R3].

Politique V1 :

- 70 % des épisodes : historique entièrement visible.
- 20 % : 1–10 % de trous dispersés.
- 10 % : un bloc manquant de taille variable.

*Ces proportions sont une proposition.* En V2, introduire des lacunes corrélées à la dynamique et des covariables incomplètes, en documentant la différence entre MCAR, MAR et MNAR.

### 9.3. Normalisation sans fuite temporelle

Pour chaque cible, calculer **sur ses valeurs d'historique visibles uniquement** :

\[
 m=\operatorname{median}(y_{1:L,obs}),\quad s=\max(1.4826\,\operatorname{MAD}(y_{1:L,obs}),\epsilon),
\]

\[
 \tilde y_t=(y_t-m)/s.
\]

`1.4826 × MAD` est une estimation robuste de l'écart type pour une distribution gaussienne, pas une garantie universelle. Pour les séries constantes ou quasi constantes : fallback explicite (p. ex. `max(std_visible, 1)` ou `scale=1` selon configuration), et journaliser son utilisation.

Les cibles futures peuvent être **transformées avec le `(m,s)` du passé**, pour calculer la loss, mais **elles ne contribuent jamais au calcul de `(m,s)`**. Idem pour past-only. Pour les known-future, définir une politique distincte de normalisation et vérifier qu'elle ne révèle aucun futur cible (une moyenne calculée sur un planning réellement connu peut être autorisée, mais ne doit pas être confondue avec un scaler appris sur les labels futurs).

**Pourquoi robust scale ?** Les outliers et discontinuités synthétiques sont voulus ; une moyenne et un écart type peuvent être sensibles à ces valeurs. La meilleure transformation (robuste, mean-scaling, asinh, etc.) sera une **ablation**, pas un dogme.

### 9.4. Quand rejeter une série générée ?

**Rejeter seulement les anomalies d'implémentation** : NaN/Inf non prévus, overflow, paramètres invalides, trajectoires impossibles au regard du générateur, non-conformité manifeste au support.

**Ne pas filtrer automatiquement** les séries peu prévisibles, très bruitées, plates, avec gros spikes ou fortes ruptures : ce sont précisément des cas nécessaires. Contrôler leur **proportion**, pas les éliminer.

---

## 10. Pipeline, API et stockage

### 10.1. Architecture recommandée du dépôt

```text
tsfm/
├── configs/
│   ├── data_v0.yaml
│   ├── data_v1.yaml
│   └── holdout_v1.yaml
├── src/data/
│   ├── generators/
│   │   ├── base.py
│   │   ├── trend_seasonal.py
│   │   ├── periodic.py
│   │   ├── gaussian_process.py
│   │   ├── autoregressive.py
│   │   ├── regime_sde.py
│   │   ├── events_counts.py
│   │   └── composite.py
│   ├── relations.py
│   ├── role_sampler.py
│   ├── episode_sampler.py
│   ├── augmentations.py
│   ├── normalizers.py
│   ├── masks.py
│   ├── dataset.py
│   ├── collate.py
│   └── validation.py
├── scripts/
│   ├── inspect_samples.py
│   ├── profile_generation.py
│   ├── build_cache.py
│   ├── build_holdouts.py
│   └── audit_dataset.py
├── tests/data/
├── reports/data/
└── docs/TSFM_DATA_REFERENCE.md
```

### 10.2. API cible : classes indépendantes de l'entraînement

```python
from dataclasses import dataclass
from typing import Any
import numpy as np

@dataclass
class SyntheticRecord:
    values: np.ndarray                  # shape [T, C]
    metadata: dict[str, Any]

class Generator:
    def sample(self, rng: np.random.Generator, length: int) -> SyntheticRecord:
        raise NotImplementedError

class EpisodeSampler:
    def sample(self, rng: np.random.Generator, record: SyntheticRecord,
               context: int, horizon: int) -> dict[str, Any]:
        raise NotImplementedError
```

En sortie du `collate_fn` PyTorch :

```python
batch = {
    "input_values": ...,      # [B, C, L+H] normalisé et masqué
    "input_mask": ...,        # [B, C, L+H]
    "target_future": ...,     # [B, C_target, H] LABELS, trainer only
    "loss_mask": ...,         # [B, C_target, H]
    "channel_roles": ...,     # [B, C]
    "scaling_stats": ...,     # pour inverser les prévisions
    "lengths": ...,
}
```

**Vérification d'interface :** le `forward()` du modèle ne reçoit pas `target_future`. Les labels sont lus seulement dans la fonction de loss. Pour des exemples à tailles variables, les tensors sont paddés et possèdent des masks explicites de padding.

### 10.3. Pipeline d'un échantillon

```text
(master_seed, split, global_sample_id)
    ↓ RNG indépendant et déterministe
sample generator family g
    ↓
sample process parameters θ + latent sources
    ↓
generate sufficiently long record (with lag buffer)
    ↓ (for multivariate) apply relations / graph
apply physical-support-safe augmentations
    ↓
sample L, H, forecast origin, target and covariate roles
    ↓
construct observation, loss and padding masks
    ↓
compute normalization using visible past ONLY
    ↓
produce batch for model; preserve labels separately
```

### 10.4. Reproductibilité

- Configuration YAML versionnée + commit Git + version des librairies.
- `master_seed` et `sample_id` composés par un générateur déterministe (`numpy.random.SeedSequence`, par exemple).
- Seeds **distinctes** pour `train`, `val_id`, `val_ood`, `test_synthetic`.
- En multi-worker, éviter `seed = global_seed + worker_id` comme seule garantie : utiliser une identité **stable par échantillon** pour qu'un changement de `num_workers` ne modifie pas les données.
- `metadata` inclut : `generator_version`, `split`, seed, famille, paramètres, roles, augmentations et origine temporelle.
- Un replay à partir d'un enregistrement de métadonnées doit régénérer les valeurs (ou fournir un hash stable de celles-ci).

### 10.5. Cache et format

- **Training simple** : streaming en RAM, pas besoin de fichier par épisode.
- **Sources coûteuses GP/SCM** : caches en shards (`.npz` compressé, Zarr, Parquet/Arrow selon accès), **pas un fichier par série** si millions d'objets.
- **Validation** : shards immuables + manifest `sha256`, commit/version/seed.
- **Métadonnées** : manifest JSONL/Parquet ; ne pas stocker de gros objets Python picklés non contrôlés.

Choisir le format en fonction des mesures de débit ; l'objectif est de ne pas laisser le GPU attendre le DataLoader.

---

## 11. Train / validation / test et prévention des fuites

### 11.1. Quatre évaluations distinctes

| Split | Origine | Réutilisable pendant le développement ? | But |
|---|---|---|---|
| `train_synthetic` | RNG entraînement | Oui | apprentissage |
| `val_id_synthetic` | nouveaux draws des familles de train | Oui | détecter overfit/dysfonctionnement |
| `val_ood_synthetic` | paramètres ou structures tenus à part | Oui | décider de généralisation hors prior immédiat |
| `test_synthetic` | sous-priors gelés avant les expériences | **Non**, sauf reporting final | conclusion expérimentale |
| `test_real` | benchmarks réels publics | **Non**, sauf reporting final si protocole strict | transfert zero-shot |

Si on utilise les benchmarks réels pour choisir régulièrement les poids du mélange, les plages ou le checkpoint, la revendication exacte devient : **weights trained synthetic-only, model selection informed by real validation**. C'est acceptable si déclaré, mais différent d'une expérimentation entièrement indépendante des données réelles.

### 11.2. Trois modes de holdout synthétique

**H1 — Nouvelles réalisations, mêmes priors :** mêmes lois de paramètres, RNG indépendant. Mesure la généralisation *in-distribution*.

**H2 — Paramètres hors support d'entraînement :** si training `P/L ∈ [0.05,0.6]`, validation sur certains ratios inédits, ou bruit/régime/durée hors plage. Ne pas étendre la validation au point où toute prédiction devient intrinsèquement impossible sans le signaler.

**H3 — Opérateurs tenus à part :** certaines compositions, graphes, lags, formes d'onde ou familles entières absentes du train. C'est le test le plus sévère, pas nécessairement une mesure de performance générale.

**Important :** des seeds différentes **ne suffisent pas** si les mêmes *records* pré-calculés sont découpés en fenêtres recouvrantes. Séparer les sources avant d'extraire les épisodes.

### 11.3. Tests anti-leakage obligatoires

```text
1. Modifier le futur de la cible -> input_values strictement identique.
2. Modifier le futur d'une covariable past-only -> input_values identique.
3. Modifier une known-future légitime -> input peut changer, cible générative reste cohérente.
4. Modifier tous les labels futurs -> scaling_stats identiques.
5. Mêmes seeds/split/config -> mêmes inputs et labels à l'octet près (tolérances si backend non déterministe).
6. Aucun record source / graphe / cache partagé entre splits.
7. Aucun exemple réel, y compris TSMixup réel, dans le DataLoader d'entraînement.
```

---

## 12. Contrôle qualité et diagnostics

### 12.1. Tests unitaires par générateur

- Forme `(T,C)` correcte, dtype, valeurs finies sauf missingness explicitement masquée.
- Reproductibilité RNG.
- Vérification support (`count >= 0`, `count entier`, valeurs saturées conformes).
- Stationnarité/instabilité cohérentes avec le label du mécanisme.
- `p_stay` et paramètres SDE appliqués comme configurés.
- GP : matrice de covariance PSD numériquement (tolérance), jitter documenté.
- Lag : l'effet relationnel apparaît à l'offset voulu dans une expérience sans bruit.
- Composition : poids/transformations présents dans metadata et reconstructibles.
- Masks : véritables zéros distinguables des manquants.

### 12.2. Rapport de distribution à générer automatiquement

Pour 10k–50k épisodes du mélange candidat, fournir :

| Statistique | Pourquoi |
|---|---|
| Proportions familles/relations/rôles | vérifier le sampler effectif |
| `std`, MAD, amplitude, quantiles, outlier rate | détecter dégénérescences |
| ACF aux retards courants | mémoire temporelle |
| Entropie/énergie spectrale | couverture de fréquences |
| Ratio puissance saisonnière/bruit | difficulté |
| Niveau de tendance et fréquence des ruptures | structure |
| Zero-rate, dispersion des comptes | intermittence |
| Fraction observée/gaps | cohérence missingness |
| Correlations, lags, causal graph depth | diversité multivariée |
| `Oracle/naive/seasonal naive` errors | estimations de prédictibilité |
| `p90/p99` du temps de génération | gestion du débit |

Les tests ADF/KPSS, ACF/FFT ou les métriques spectrales sont **diagnostiques** : ne pas les prendre comme une vérité absolue sur des séries courtes ou non stationnaires.

### 12.3. Définir une difficulté utile

Sur des holdouts synthétiques, comparer des références :

- **Last-value naive**.
- **Seasonal naive** (quand période définie et observable).
- **Linear extrapolation / drift**.
- **Oracle conditionnel** quand disponible (connaît la famille et éventuellement les paramètres, **pas les innovations futures**).

Certaines familles stochastiques ont une erreur irréductible importante. Un mauvais score absolu ne signifie pas forcément une mauvaise architecture ; l'écart à une référence pertinente est plus instructif.

### 12.4. Critères de qualité minimaux avant training

- 0 violation des contrats de shapes/masks/roles.
- 0 fuite identifiée par tests d'invariance.
- 0 overflow/NaN non prévu.
- Répartition par famille proche de la configuration (tolérance statistique).
- Présence effective de cas simples, intermédiaires et imprévisibles.
- Les relations utiles montrent un *covariate lift* mesurable avec des modèles diagnostiques ; les decoys ne montrent pas de lift systématique.
- Pipeline reproductible avec seed et version.
- Débit mesuré sur le hardware cible.

---

## 13. Comment savoir si les données sont suffisantes ?

Ne pas décider « nous avons assez de données » parce que 1M ou 10M est atteint. Décider sur **courbes et diagnostics**.

### 13.1. Tableau de bord obligatoire

Pour chaque checkpoint :

- Loss train et validation ID.
- Quantile/pinball loss OOD par famille et horizon ; point loss pour le prototype.
- Calibration : couverture empirique des intervalles à 50 %/80 %/90 % (pour un modèle probabiliste).
- Skill vs seasonal naive et simple baselines.
- Performances sur `H/L` élevé, ruptures de régimes, spikes, GP, données manquantes.
- Différence avec/sans covariables sur tâches relationnelles utiles, faibles et decoy.
- Coût en heure GPU et temps génération par mille épisodes.

### 13.2. Arbre de décision

```text
OOD stagne ?
├── Train stagne aussi → vérifier capacité, loss, learning rate, pipeline, compute.
├── Train progresse / OOD régresse → sur-spécialisation au prior ; diversifier.
├── Seules SDE/ruptures échouent → enrichir G5/G6, pas toutes les familles.
├── Long horizons échouent → adapter P/L, H/L et historique observé.
├── Avec covariables ≈ sans covariables → contrôler pertinence, rôle, masque, relation.
└── ID+OOD progressent → augmenter budget avant de tout refondre.
```

**Seuil de plateau proposé :** si, sur deux augmentations substantielles du budget, le gain OOD reste inférieur à ~1–2 % relatif avec une incertitude statistique couvrant zéro, considérer prioritairement l'extension de diversité. Ce seuil est une **règle de gestion proposée**, pas un résultat de paper.

---

## 14. Expériences d'ablation

### 14.1. Principe de comparaison propre

Pour isoler **la donnée**, conserver : architecture, nombre de paramètres, budget de mises à jour, distribution `(L,H,C)`, optimizer, seeds (si possible), métriques et validation.

**Attention :** retirer une famille puis redistribuer son poids aux autres change plus d'une chose. Documenter explicitement la redistribution et éventuellement comparer une variante où son poids est remplacé par un générateur contrôle.

### 14.2. Matrice initiale

| Expérience | Contrôle | Variante | Question |
|---|---|---|---|
| E1 | Mélange complet | Sans G3 GP | Lissage / structures non paramétriques |
| E2 | Mélange complet | Sans G5 SDE | Robustesse aux régimes/volatilité |
| E3 | Mélange complet | Sans G6 events | Ruptures et intermittence |
| E4 | Mélange complet | Sans G7 composite | Composition vs primitives simples |
| E5 | Mix complet | GP seul / tendance seule | Intérêt de la diversité |
| E6 | Mix complet | Poids uniformes | Poids proposés utiles ? |
| E7 | Mix complet | Volumes 0.2M / 1M / 3M | Courbe de scaling des exemples |
| E8 | Multivarié | Sans decoys | Robustesse aux covariables inutiles |
| E9 | Multivarié | Sans lags ou shared drivers | Compréhension relationnelle |
| E10 | Curriculum | Distribution multivariée fixe | Effet du curriculum |
| E11 | Missingness | Sans missingness | Robustesse / coût sur cas complets |
| E12 | Normalisation robuste | Mean/std, asinh, autre | Stabilité / calibration |

**Plan de calcul :** première passe sur 5M avec budget raccourci ; trois seeds pour les expériences prometteuses et intervalles de confiance (bootstrap par source/tâche, pas par fenêtres très corrélées).

---

## 15. Recette de démarrage et configuration YAML

### 15.1. Recette V0 — Implémentable immédiatement

1. Implémenter G1, G2, G4, G5, G6 ; laisser G3/G7 comme extensions contrôlées si GP trop lent.
2. Générer **10k sources** avec seeds déterministes ; produire des figures et le rapport QA.
3. Geler `val_id` et `val_ood` (proposer **5k épisodes chacun**), sans recouvrement de sources.
4. Entraîner un **prototype de 5M paramètres** sur `20k–100k épisodes vus` ; prouver la capacité d'overfit sur 32 exemples fixes.
5. Activer GP + composites ; refaire QA et tests anti-leakage.
6. Piloter `200k–500k épisodes` ; tracer loss OOD par mécanisme.
7. Ajouter relations, rôles et decoys ; faire le test `avec/sans covariables`.
8. Ne lancer l'échelle 25–28M qu'une fois pipeline et benchmarks synthétiques fiables.

### 15.2. Exemple de configuration de départ

```yaml
schema_version: 1
seed: 20261107
training_data: synthetic_only

stage: pilot_univariate
sequence:
  context_lengths: {128: 0.15, 256: 0.25, 512: 0.40, 1024: 0.20}
  horizons:        {24: 0.20, 48: 0.25, 96: 0.35, 192: 0.20}
  patch_size: 32
  min_history_observed: 32

generator_mix:
  trend_seasonal: 0.15
  complex_periodic: 0.10
  gaussian_process: 0.15
  autoregressive: 0.15
  regime_sde: 0.20
  events_counts: 0.10
  composite: 0.15

univariate_generators:
  trend_seasonal:
    harmonics: [0, 1, 2, 3]
    seasonal_periods_steps: [7, 24, 48, 96, 168, 365]
  autoregressive:
    ar1_phi_range: [-0.95, 0.95]
  gaussian_process:
    kernels: [rbf, matern, periodic, linear, rational_quadratic, white]
    max_exact_length: 512
    max_kernels: 4
    cache_if_longer: true
  regime_sde:
    regime_stay_probability_range: [0.90, 0.999]

augmentations:
  probability_any: 0.45
  missingness:
    complete: 0.70
    sparse: 0.20
    block: 0.10
  magnitude_scale: true
  random_offset: true
  sign_flip_only_when_semantically_valid: true

multivariate:
  enabled: false
  max_channels: 1
  relational_primitives: [signed_mix, lag, nonlinear, shared_driver,
                          event_effect, cointegration, decoy]
  known_future_allowed_types: [calendar, planned_event, scheduled_promotion]

normalization:
  method: median_mad
  statistics_window: visible_history_only
  fallback_scale: 1.0

sampling:
  on_the_fly: true
  cache_expensive_generators: true
  disjoint_seed_namespaces: [train, val_id, val_ood, test_syn]

validation:
  val_id_episodes: 5000
  val_ood_episodes: 5000
  freeze_manifest: true
```

**Attention à l'implémentation :** `max_exact_length=512` ne signifie pas « tronquer une série de 1024 et l'appeler GP exact ». Si `L+H > 512`, utiliser un générateur GP approximatif documenté, un autre algorithme performant, ou re-échantillonner une autre famille tout en **corrigeant les probabilités effectives et en les rapportant**.

### 15.3. Exemple de passage à l'étape multivariée

Dans une configuration dérivée :

```yaml
stage: multivariate_pilot
multivariate:
  enabled: true
  min_channels: 2
  max_channels: 4
  max_targets: 2
  minimum_decoy_fraction_of_tasks: 0.15
curriculum:
  # Fraction d'épisodes multivariés pendant l'entraînement : proposition.
  milestones:
    - {progress: 0.0, multivariate_probability: 0.10}
    - {progress: 0.5, multivariate_probability: 0.20}
    - {progress: 1.0, multivariate_probability: 0.40}
```

Cette progression est inspirée du principe du curriculum de t0-alpha, mais les *milestones* par fractions de progression et nos quotas ne reproduisent pas ses steps exactement [R4].

---

## 16. Plan d'implémentation et critères d'acceptation

### Sprint D1 — Infrastructure, 2–3 jours

**Tâches :** contrat d'épisode, RNG déterministe, split, masks, normalizer, tests unitaires.  
**Fini lorsque :** test d'invariance au futur + test de replay + test de zéro observé vs missing passent.

### Sprint D2 — Générateurs rapides, 3–5 jours

**Tâches :** G1, G2, G4, G6 ; visualisation et stats.  
**Fini lorsque :** 1k séries/famille auditables ; mesure de `series/sec` ; oracle/baselines plausibles.

### Sprint D3 — Processus complexes, 4–7 jours

**Tâches :** G3 (GP exact puis approximation), G5 (SDE), G7 (composites) ; caches.  
**Fini lorsque :** pas de dysfonctionnements numériques et débit acceptable, distributions non dégénérées.

### Sprint D4 — Mix, validation, QA, 2–4 jours

**Tâches :** sampler pondéré, paliers d'épisodes, holdouts ID/OOD, rapport automatique.  
**Fini lorsque :** pilot training reproductible ; profils enregistrés et val gelée.

### Sprint D5 — Relations, 4–7 jours

**Tâches :** graphes, lags, rôles, known-future, decoys, masks multivariés.  
**Fini lorsque :** relation utile mesurable, decoy inutile, aucune fuite future, `C` variable.

### Sprint D6 — Expériences quantitatives, 1 semaine ou selon compute

**Tâches :** volume scaling, ablations G3/G5/G6, premiers scores probabilistes.  
**Fini lorsque :** on sait *pourquoi* notre mélange est préférable à des variantes simples, avec courbes et incertitude.

---

## 17. FAQ / décisions justifiées

**Q : Pourquoi ne pas générer directement 10 millions de séries ?**  
**R :** Parce que le chiffre seul ne dit rien sur la diversité, la réutilisation ni les gains marginaux. Commencer par la stabilité et les courbes de scaling ; TempoPFN fournit un ordre de grandeur, pas un minimum universel [R2].

**Q : Pourquoi sept familles et pas 30 ?**  
**R :** Le nombre de familles est arbitraire ; la couverture des **propriétés structurelles** importe plus. Les sept familles sont un compromis d'implémentation. Ajouter une famille doit être justifié par un axe actuellement absent ou une ablation positive.

**Q : Pourquoi les Gaussian Processes si c'est lent ?**  
**R :** Les kernels compositionnels produisent des structures lisses/irrégulières et des corrélations non triviales ; exact sampling à petite longueur, approximations/cache ensuite [R6, R7].

**Q : Pourquoi autant de SDE et de ruptures ?**  
**R :** TempoPFN observe un effet important des SDE sur son benchmark. Notre poids de 20 % est une **hypothèse**, à comparer à 10 % et à une pondération uniforme [R2].

**Q : Pourquoi générer du bruit blanc et des séries impossibles à prévoir ?**  
**R :** Pour apprendre que certaines incertitudes sont irréductibles et éviter des intervalles trop étroits. Ne pas en mettre assez peut induire une surconfiance ; trop en mettre peut rendre le modèle paresseux [R5].

**Q : Doit-on conserver toutes les séries en mémoire ?**  
**R :** Non. Génération streaming pour les familles rapides ; cache pour GP/SCM ; splits synthétiques de validation gelés. Profiler avant de choisir le format.

**Q : Peut-on mixer des séries synthétiques entre elles ?**  
**R :** Oui, avec un mélange contrôlé, des séries alignées et une normalisation adaptée. Ne pas prétendre reproduire TSMixup de Chronos si l'on change la provenance ou la procédure [R6].

**Q : Comment gérer les séries de 100 ou 10 000 observations ?**  
**R :** Échantillonner `L` et `H` ; utiliser un pool de longueurs ; allonger seulement après avoir validé coût et robustesse. Les périodes doivent être ajustées à l'échelle disponible.

**Q : Comment choisir l'importance d'un générateur ?**  
**R :** Pas à partir de son nom ou de ses belles courbes. Vérifier sa couverture statistique, son coût et l'ablation à **budget fixé**, par famille/horizon/OOD.

**Q : Comment prouver que le modèle exploite les covariables ?**  
**R :** Évaluer à checkpoint constant avec/sans covariable, covariables décalées, permutées et decoys, sur tâches où l'effet génératif est connu [R3, R4].

**Q : Les covariables futures synthétiques peuvent-elles être parfaitement connues ?**  
**R :** Oui, uniquement si c'est cohérent avec le scénario (calendrier, planning fixé). Sinon créer une variable de prévision **bruitée** disponible ex ante et cacher le vrai futur du processus latent.

**Q : Le fait que le modèle soit sous 30M change-t-il la génération ?**  
**R :** Pas la logique causale, mais le budget de préentraînement : éviter un mélange si complexe que le petit modèle n'apprend aucun motif ; mesurer une progression `5M → 25–28M` sur **le même protocole de données**.

**Q : Comment éviter le data leakage avec la normalisation ?**  
**R :** Toutes les statistiques de cible et de past-only doivent dépendre uniquement de valeurs effectivement disponibles avant le cutoff. Tester l'invariance des scalers lorsque les labels futurs changent.

**Q : Peut-on dire « state of the art grâce à nos données synthétiques » ?**  
**R :** Seulement si les résultats, comparaisons, budgets et protocoles le démontrent. Timestral, Timer-M1 et t0 mélangent, à divers stades, des données réelles et synthétiques ; leurs performances publiées ne sont pas des preuves directes pour notre protocole 100 % synthétique [R3–R5].

---

## 18. Bibliographie commentée

> Les numéros `[R#]` cités dans ce document désignent des **sources primaires**, consultées le **7 novembre 2026**. Les valeurs chiffrées non attribuées explicitement à une source sont des **propositions de conception**.

- **[R1]** Dooley et al. (2023). *ForecastPFN: Synthetically-Trained Zero-Shot Forecasting*. NeurIPS. [Article](https://arxiv.org/abs/2311.01933) · [Code](https://github.com/abacusai/ForecastPFN). **À lire :** prior synthétique de séries et formulation zero-shot.
- **[R2]** Moroshan et al. (version arXiv du 2 novembre 2026). *TempoPFN: Synthetic Pre-training of Linear RNNs for Zero-shot Time Series Forecasting*. [Article](https://arxiv.org/abs/2510.25502) · [Code](https://github.com/automl/TempoPFN). **À lire :** §3.2–3.3, §4.3–4.4, annexes C–D. Contient les générateurs, l'ablation SDE, ~10M séries et le profilage de débit.
- **[R3]** Zhang et al. (2026). *Timer-M1: A Multivariate Time Series Foundation Model via Learning Primitives*. [Article](https://arxiv.org/abs/2610.11734). **À lire :** §3.1, annexe A, ablation *Synthetic Only* vs *Full Recipe*. Distingue primitives temporelles et relationnelles, puis rôles/masques d'épisodes.
- **[R4]** Meyer et al. (2026). *t0: A Time-Series Foundation Model for Forecasting with Context*. [Rapport](https://arxiv.org/abs/2609.24559) · [Code](https://github.com/theforecastingcompany/tfc-t0). **À lire :** §2, §4.2. Les informations détaillées concernent principalement **t0-alpha/beta**, pas le régime de données de t0-gamma.
- **[R5]** Petralia et al. (2026). *Timestral*. [Article](https://arxiv.org/abs/2610.37585) · [Présentation](https://mistral.ai/news/timestral/). **À lire :** §2.1, §4.1 : prior de prédiction, synthétique uni-/multivarié, événements/régimes, puis post-training réel + synthétique.
- **[R6]** Ansari et al. (2024). *Chronos: Learning the Language of Time Series*. TMLR. [Article](https://arxiv.org/abs/2403.07815) · [Code](https://github.com/amazon-science/chronos-forecasting). **À lire :** §4 (KernelSynth et TSMixup), annexe A (pseudo-code).
- **[R7]** Rahimi & Recht (2007). *Random Features for Large-Scale Kernel Machines*. NeurIPS. [Publication](https://proceedings.neurips.cc/paper_files/paper/2007/hash/013a006f03dbc5392effeb8f18fda755-Abstract.html). **À lire :** approximation de kernels via random features ; piste pour réduire le coût des GP, **à valider** pour le générateur exact choisi.
- **[R8]** Salesforce AI Research. *GIFT-Eval*. [Benchmark / code](https://github.com/SalesforceAIResearch/gift-eval). **Usage :** évaluation réelle finale, pas training.
- **[R9]** AutoGluon. *fev — Forecast Evaluation library*. [Code](https://github.com/autogluon/fev). **Usage :** tâches et évaluation reproductibles, y compris covariables et protocole de scoring.

---

## Résumé des décisions à acter avant de coder

- [ ] Geler la définition exacte de « entraînement 100 % synthétique ».
- [ ] Valider le contrat d'épisode et les trois rôles de canaux.
- [ ] Implémenter les tests de masques **avant** le premier Transformer.
- [ ] Approuver le mélange initial des sept familles comme **point de départ expérimental**.
- [ ] Préparer des holdouts synthétiques ID, paramètres OOD et mécanismes OOD.
- [ ] Fixer le budget du **premier pilote** et enregistrer les expositions, pas seulement les fichiers créés.
- [ ] Mesurer le débit et décider du cache GP sur chiffres réels.
- [ ] Écrire les tests de scaling sur historique visible et de no-future-leakage.
- [ ] Geler les benchmarks réels pour l'évaluation finale.
- [ ] Choisir les trois premières ablations (recommandées : sans SDE, sans GP, sans composites).

**Critère de réussite de la partie données :** le pipeline fabrique à la demande des épisodes variés, **causalement cohérents**, reproductibles, sans fuite d'information ; le modèle progresse sur plusieurs familles et sur des holdouts synthétiques OOD ; les résultats peuvent être reliés à des choix mesurables sur les générateurs, leurs mélanges et leurs volumes.
