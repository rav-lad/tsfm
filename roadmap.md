# Roadmap 

Avec **Timestral (Mistral AI), Timer-M1 (Tsinghua University) et t0-gamma (The Forecasting Company)**, je modifierais légèrement notre stratégie initiale.

Ces modèles montrent trois directions particulièrement intéressantes : apprendre à partir de données synthétiques diversifiées, modéliser explicitement les relations entre séries et prédire plusieurs horizons avec une sortie probabiliste.

## 1. Ce que nous devons retenir des trois modèles

| Modèle | Taille | Architecture | Données d'entraînement | Innovation à retenir |
|---|---:|---|---|---|
| **Timestral** (octobre 2026) | 5M, 28M, 123M, 209M | Transformer causal à patches | Préentraînement synthétique, puis post-training réel + synthétique | Decoupled Horizon Forecasting (DHF) |
| **Timer-M1** (octobre 2026) | 121M | Transformer 2D avec attention intervariables contrôlée par des gates | Réel + synthétique | Temporal & Relational Primitives |
| **t0-gamma** (octobre 2026) | ~256M | Patch Transformer avec attention temporelle et intervariables | Détails complets du training de gamma non publiés dans le rapport initial de t0 | Forecasting avec covariables passées et futures |

Sources : articles scientifiques de Timestral, Timer-M1 et t0, ainsi que la documentation de t0-gamma. [arXiv](https://arxiv.org/html/2610.37585?utm_source=chatgpt.com)

### Pourquoi Timestral est particulièrement intéressant

Mistral a publié un modèle de **27,8 millions de paramètres**, exactement dans notre budget.

Sa configuration :

| Hyperparamètre | Timestral-28M |
|---|---:|
| Transformer blocks | 12 |
| Hidden dimension | 384 |
| Attention heads | 6 |
| Head dimension | 64 |
| SwiGLU hidden width | 1024 |
| Patch size | 32 |
| Quantiles prédits | 21 |
| Paramètres | 27,8M |

Les auteurs rapportent que cette version dépasse Chronos-2 sur GIFT-Eval et TIME, malgré une taille sensiblement inférieure. [arXiv](https://arxiv.org/html/2610.37585?utm_source=chatgpt.com)

Cela change notre objectif : plutôt que de chercher une architecture totalement nouvelle, nous pouvons construire **notre propre petit modèle inspiré de Timestral, tout en expérimentant avec les méthodes de génération de Timer-M1 et de t0**.

La distinction importante est que notre modèle restera entraîné exclusivement sur des données synthétiques. Les résultats des modèles publiés, qui utilisent aussi des données réelles, ne seront donc pas directement reproductibles avec notre protocole.

---

## 2. Les trois idées que je reprendrais

### A. Timestral : Decoupled Horizon Forecasting

C'est l'innovation architecturale que je trouve la plus intéressante pour notre projet.

Traditionnellement, un Transformer causal prédit le prochain patch, puis réutilise ses prédictions pour anticiper les patches suivants.

Le problème est que les erreurs peuvent s'accumuler au fil du forecasting.

Timestral propose une autre approche, le **Decoupled Horizon Forecasting (DHF)**.

Le principe :

1. Les observations historiques sont encodées causalement.
2. Des *prediction tokens* distincts représentent les patches futurs.
3. Ces tokens peuvent accéder au contexte autorisé.
4. Le modèle prédit directement plusieurs patches futurs sans réinjecter ses prédictions intermédiaires.

L'avantage supplémentaire du DHF concerne l'entraînement : les observations utilisées comme cibles à un instant peuvent rester disponibles comme contexte pour les prédictions démarrant plus tard. On évite ainsi de sacrifier systématiquement du contexte pour entraîner les horizons longs.

Timestral peut produire jusqu'à 1 024 pas de forecasting dans un même bloc de prédiction. [arXiv](https://arxiv.org/html/2610.37585?utm_source=chatgpt.com)

**Pour notre modèle :** nous implémenterons d'abord un direct multi-horizon forecasting simple, puis une version DHF pour mesurer si elle améliore les prédictions.

### B. Timer-M1 : Temporal & Relational Primitives

Timer-M1 propose une manière particulièrement intéressante d'organiser la génération des données.

Il distingue deux types de mécanismes :

**Temporal primitives :** les dynamiques propres à chaque série.

Par exemple, tendances, périodicités, bruit autocorrélé et processus stochastiques.

**Relational primitives :** les relations entre plusieurs séries.

Par exemple :

\[
x_1(t)=\sin(t)+\epsilon_1(t)
\]

\[
x_2(t)=0.7x_1(t-5)+\epsilon_2(t)
\]

Ici, \(x_2\) dépend de \(x_1\), avec un décalage temporel.

Timer-M1 utilise des générateurs de relations linéaires, non linéaires, retardées et de facteurs communs, avec une architecture qui distingue l'attention temporelle de l'attention intervariables.

Son étude d'ablation indique qu'un modèle entraîné uniquement sur ses données synthétiques obtient un skill score FEV de 0,4710, contre 0,4887 pour la recette complète combinant données réelles et synthétiques. [arXiv](https://arxiv.org/html/2610.11734?utm_source=chatgpt.com)

C'est encourageant pour notre projet, même si le modèle uniquement synthétique reste moins performant dans cette comparaison.

**Pour notre modèle :** je reprendrais surtout le principe de génération par primitives et de composition entre séries.

### C. t0-gamma : apprendre à utiliser des covariables

La famille t0 est particulièrement intéressante pour apprendre à prévoir une variable à partir d'informations extérieures.

Prenons un exemple :

Nous souhaitons prédire une consommation électrique \(y_t\).

Nous possédons également :

- La consommation historique.
- La température.
- Le jour de la semaine.
- Les jours fériés.
- Une prévision de température pour les prochains jours.

Le modèle doit apprendre que certaines informations futures sont autorisées, mais que la consommation future reste inconnue.

Dans le rapport scientifique de t0, les auteurs utilisent notamment des **structural causal models**, des processus gaussiens et des générateurs d'effets entre covariables et cibles.

Ils introduisent aussi des *decoy covariates*, c'est-à-dire des variables qui ne contiennent aucune information utile pour la cible.

L'objectif est d'entraîner un modèle capable d'exploiter les relations prédictives sans dépendre de variables non pertinentes. [arXiv](https://arxiv.org/pdf/2609.24559?utm_source=chatgpt.com)

Les résultats publiés pour t0-gamma en octobre 2026 sont notamment :

| Benchmark | Résultat annoncé |
|---|---:|
| GIFT-Eval CRPS | 0,4520 |
| TIME CRPS | 0,5340 |
| fev-bench Skill | 49,42 % |

Il s'agissait des résultats annoncés lors de sa sortie, le 13 octobre 2026. Timestral a ensuite publié de nouveaux résultats compétitifs, donc ces classements ne doivent pas être considérés comme permanents. [The Forecasting Company](https://www.theforecastingcompany.com/blog/t0-gamma/?utm_source=chatgpt.com)

---

# 3. Nouvelle Phase 1 — Synthetic Data Generation

Je conserverais les sept familles de générateurs proposées dans la roadmap initiale.

Mais j'ajouterais un second niveau inspiré de Timer-M1 et t0 : **la génération synthétique multivariée**.

Nous aurons donc deux composants.

### Composant 1 : Univariate Generator

Il produira des processus indépendants :

\[
x_t=G_{\theta}(t,\epsilon)
\]

Avec nos familles :

- Trends et saisonnalités.
- Gaussian Processes.
- Autoregressive processes.
- Regime-switching SDEs.
- Complex periodicity.
- Intermittent events.
- Composite nonlinear processes.

Pour cette partie, Chronos KernelSynth et TempoPFN restent nos deux principales références, car ils proposent des mécanismes synthétiques directement utiles à notre objectif. [GitHub](https://github.com/amazon-science/chronos-forecasting/blob/main/scripts/README.md?utm_source=chatgpt.com)

### Composant 2 : Multivariate Relational Generator

Nous combinons ensuite les séries pour créer des dépendances :

\[
X_t=
\begin{bmatrix}
x_1(t)\\
x_2(t)\\
\vdots\\
x_C(t)
\end{bmatrix}
\]

Avec des relations comme :

\[
x_3(t)=
a\,x_1(t-\tau_1)
+b\,x_2(t-\tau_2)
+\epsilon_3(t)
\]

Et éventuellement :

\[
x_4(t)=
\tanh\left(
w_1x_1(t)+w_2x_2(t)
\right)+\epsilon_4(t)
\]

J'ajouterais cinq types de relations :

| Relation | Inspiration |
|---|---|
| Linear dependencies | Timer-M1 |
| Nonlinear dependencies | Timer-M1 |
| Lagged effects | Timer-M1 / t0 |
| Shared latent drivers | t0 / Timer-M1 |
| Causal graphs avec variables inutiles | t0 |

Nous pourrons aussi générer des covariables futures connues, comme des calendriers, des événements planifiés ou des signaux exogènes simulés.

**C'est probablement là que notre projet deviendra le plus intéressant.**

Au lieu de produire uniquement des séries isolées, nous pourrons créer des problèmes de forecasting où le modèle doit découvrir les relations pertinentes entre variables.

### Livrable

Un générateur qui produit des *forecasting episodes* :

```python
{
    "target_history": ...,
    "past_covariates": ...,
    "future_covariates": ...,
    "target_future": ...,
    "observation_mask": ...,
    "metadata": ...
}
```

Les `metadata` décriront les mécanismes synthétiques pour nos tests, mais ne seront pas données au modèle pendant le forecasting.

---

# 4. Nouvelle Phase 2 — Architecture du modèle

Je propose deux versions successives.

| | Prototype | Modèle final |
|---|---|---|
| Paramètres cibles | ~5M | 25–28M |
| Hidden dimension | 192 | 384 |
| Transformer layers | 8 | 12 |
| Attention heads | 3 | 6 |
| Patch size | 32 | 32 |
| Forecasting | Direct multi-horizon | Direct, puis DHF |
| Multivariate | Non | Oui |
| Quantile head | 5 quantiles | Jusqu'à 21 quantiles |

Ces configurations sont inspirées de Timestral-5M et Timestral-28M. Le nombre exact de paramètres de notre modèle dépendra des modules que nous implémenterons.

### Architecture proposée

```text
                    INPUT
                      |
             Context Normalization
                      |
                Patch Embedding
                  P = 32
                      |
              Temporal Attention
                      |
                   SwiGLU
                      |
              Temporal Attention
                      |
                   SwiGLU
                      |
               Variate Attention
                (avec gating)
                      |
                    ...
                      |
             Prediction Tokens
                      |
               Quantile Head
                      |
              Denormalization
                      |
                   OUTPUT
```

Ce schéma représente l'architecture cible, pas nécessairement l'ordre exact de tous les blocs.

Je retiendrais :

**De Timestral :** les prediction tokens, le DHF, RMSNorm, SwiGLU, les quantiles et les patches de 32 observations.

**De Timer-M1 :** l'attention intervariables avec des gates apprises, pour contrôler la contribution des autres séries.

**De t0 :** une distinction explicite entre séries cibles, covariables historiques et covariables futures connues.

La gestion des masques d'attention sera critique. Nous devrons notamment garantir qu'aucune information future sur les cibles ne peut revenir indirectement à travers une covariable.

### Pourquoi ne pas tout implémenter immédiatement ?

Parce que nous devons pouvoir comprendre d'où viennent les gains.

Nous procéderons par étapes :

| Version | Fonctionnalités |
|---|---|
| V0 | Univariate Patch Transformer, prédiction ponctuelle |
| V1 | Quantile forecasting |
| V2 | Direct multi-horizon + DHF |
| V3 | Multivariate forecasting |
| V4 | Covariate-aware forecasting |

Cette progression permettra de comparer chaque innovation avec la version précédente.

---

# 5. Nouvelle Phase 3 — Training

Pour l'entraînement, je combinerais les idées de Timestral et du curriculum de t0.

## Étape A — Sanity checks

Avant le préentraînement, nous vérifierons que le modèle peut apprendre quelques problèmes simples.

Par exemple :

- Prédire une sinusoïde avec une période fixe.
- Extrapoler une tendance linéaire.
- Prédire un processus AR(1).
- Exploiter une covariable décalée.

Le modèle devra aussi être capable de surapprendre un petit batch fixe.

C'est indispensable pour détecter les erreurs dans les masks, la normalisation ou la fonction de loss.

## Étape B — Univariate pretraining

Nous entraînerons le modèle sur les sept familles de générateurs.

Je commencerais avec :

- Context lengths : 128, 256, 512, 1024.
- Horizons : 32, 64, 96, 192.
- Une loss de prédiction ponctuelle pour le prototype.
- Puis une pinball loss sur plusieurs quantiles.

La pinball loss est également utilisée par Timestral et Timer-M1 :

\[
\mathcal{L}_{\tau}(y,\hat{q}_{\tau})
=
\max\left(
\tau(y-\hat{q}_{\tau}),
(\tau-1)(y-\hat{q}_{\tau})
\right)
\]

Elle permet d'apprendre les quantiles de la distribution prédictive plutôt qu'une seule estimation centrale. [arXiv](https://arxiv.org/html/2610.37585?utm_source=chatgpt.com)

Les longueurs proposées sont des choix de départ adaptés au prototypage, et non la recette exacte d'un de ces modèles.

## Étape C — Multivariate curriculum

Une fois le forecasting univarié fonctionnel, nous introduirons progressivement les relations entre variables.

C'est inspiré de la stratégie utilisée pour t0-alpha, où la proportion d'exemples multivariés augmente au cours du préentraînement. [arXiv](https://arxiv.org/pdf/2609.24559?utm_source=chatgpt.com)

Je propose le curriculum suivant :

| Étape | Univariate | Multivariate |
|---|---:|---:|
| Début | 90 % | 10 % |
| Milieu | 70 % | 30 % |
| Fin | 50 % | 50 % |

Ces proportions seront expérimentales.

Nous pourrons ensuite évaluer si ce curriculum fonctionne mieux qu'un entraînement utilisant dès le début une proportion fixe de données multivariées.

## Étape D — Long-context training

Quand les modèles plus petits convergeront correctement, nous augmenterons les context lengths.

Nous commencerons par 512–1024 observations avant d'essayer 2048, puis éventuellement 4096.

**Je ne chercherais pas immédiatement à reproduire les 8192 observations du préentraînement Timestral.**

Le coût de l'attention, le nombre de variables et la taille des batchs risquent de devenir les principaux facteurs limitants.

Il faudra mesurer les performances avant d'augmenter le contexte.

### Concernant le matériel

Avec une GTX 1660 de 6 Go, je limiterais le GPU local aux tests, au débogage et aux premiers entraînements d'un petit modèle.

Un modèle de 28M paramètres peut sembler léger, mais l'entraînement nécessite de stocker les gradients, les états de l'optimiseur et les activations, souvent bien plus coûteuses en mémoire que les paramètres eux-mêmes.

Pour le préentraînement complet, je privilégierais un GPU cloud disposant idéalement de 24 Go de VRAM ou davantage.

Le coût exact dépendra surtout du nombre d'exemples, de la longueur des séquences, du nombre de variables et de l'efficacité de notre implémentation.

---

# 6. Phase 4 — Evaluation et expériences scientifiques

C'est une partie importante si tu veux produire autre chose qu'une simple reproduction technique.

Je testerais le modèle sur trois benchmarks reconnus :

| Benchmark | Ce qu'il permet d'évaluer |
|---|---|
| GIFT-Eval | Généralisation sur différents domaines et horizons |
| TIME | Zero-shot forecasting sur de nombreuses tâches |
| fev-bench | Prévision avec ou sans covariables, selon les tâches |

Nous comparerons nos résultats avec des modèles classiques et plusieurs TSFM.

Les comparaisons devront rester équitables : même historique disponible, mêmes covariables autorisées, mêmes métriques et mêmes tâches.

Il faudra également comparer notre modèle à Timestral-5M et Timestral-28M. Ce sont des références plus pertinentes pour juger l'efficacité de notre architecture que t0-gamma, qui est beaucoup plus grand.

## Les ablations que je considère indispensables

| Expérience | Question de recherche |
|---|---|
| Sans Gaussian Processes | KernelSynth améliore-t-il la généralisation ? |
| Sans SDE | Les processus à changements de régime sont-ils essentiels ? |
| Sans causal graphs | Les dépendances structurées améliorent-elles les résultats multivariés ? |
| Sans relational primitives | Le modèle exploite-t-il réellement les autres séries ? |
| Direct forecasting vs DHF | Le DHF améliore-t-il les horizons longs ? |
| Avec / sans variate attention | L'attention multivariée apporte-t-elle un gain ? |
| 5M vs 28M | Dans quelle mesure les performances évoluent-elles avec la taille ? |

Une expérience que j'aimerais particulièrement ajouter est **une évaluation sur des générateurs synthétiques jamais rencontrés pendant l'entraînement**.

Par exemple, nous pouvons entraîner sur les familles A à F, puis tester sur une famille G.

Cela permet d'évaluer si le modèle a appris des mécanismes transférables ou simplement mémorisé les distributions de nos générateurs.

Il faudra aussi garder les benchmarks réels hors du processus d'entraînement et éviter d'ajuster indéfiniment nos générateurs en fonction des scores de test.

---

# 7. Planning final du projet

| Semaine | Travail | Livrable |
|---|---|---|
| 1 | Implémenter les générateurs univariés | Synthetic Generator V1 |
| 2 | Gaussian Processes, SDE et compositions | Synthetic Generator V2 |
| 3 | Relations multivariées et causal graphs | Multivariate Generator |
| 4 | Implémenter le Patch Transformer ~5M | Model V0 |
| 5 | Quantile head et forecasting multi-horizon | Model V1 |
| 6 | Premier préentraînement et corrections | Checkpoint 5M |
| 7 | DHF et variate attention | Architecture finale |
| 8 | Préentraînement du modèle 25–28M | Checkpoint final |
| 9 | Benchmarking | Rapport de performances |
| 10 | Ablations et documentation | GitHub + rapport technique |

Ce planning suppose que les entraînements restent relativement courts. Il faudra éventuellement ajouter du temps de calcul et d'expérimentation.

## 8. Les papers prioritaires à lire

Je réduirais notre bibliographie de départ à cinq papers principaux.

| Priorité | Paper | Ce que tu dois étudier |
|---|---|---|
| **1** | [Timestral (2026)](https://arxiv.org/abs/2610.37585?utm_source=chatgpt.com) | Architecture, DHF, quantile head, synthetic pretraining |
| **2** | [Timer-M1 (2026)](https://arxiv.org/abs/2610.11734?utm_source=chatgpt.com) | Temporal primitives, relational primitives, gated attention |
| **3** | [t0: A Time-Series Foundation Model for Forecasting with Context (2026)](https://arxiv.org/abs/2609.24559?utm_source=chatgpt.com) | Causal graphs, covariables, curriculum |
| **4** | [TempoPFN (2025/2026)](https://arxiv.org/abs/2510.25502?utm_source=chatgpt.com) | Entraînement exclusivement synthétique, SDE, ablations |
| **5** | [Chronos (2024)](https://arxiv.org/abs/2403.07815?utm_source=chatgpt.com) | KernelSynth, Gaussian Processes |

Les repositories de [Timestral](https://github.com/mistralai/timestral?utm_source=chatgpt.com), [t0](https://github.com/theforecastingcompany/tfc-t0?utm_source=chatgpt.com) et [TempoPFN](https://github.com/automl/TempoPFN?utm_source=chatgpt.com) seront particulièrement utiles pour vérifier les détails d'implémentation.

---

## 9. L'orientation scientifique que je choisirais

Je formulerais le projet autour de cette question :

> **How far can a sub-30M parameter Time Series Foundation Model generalize to real-world forecasting tasks when trained exclusively on compositional synthetic data?**

Avec trois axes expérimentaux :

**1. Data diversity.** Mesurer l'influence de la diversité des générateurs synthétiques sur les performances zero-shot.

**2. Relational learning.** Étudier si l'entraînement sur des relations causales synthétiques permet d'exploiter efficacement les covariables réelles.

**3. Architecture efficiency.** Comparer direct multi-horizon forecasting, DHF et attention intervariables sous un budget de paramètres limité.

