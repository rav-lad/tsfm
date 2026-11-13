# Synthetic data

No real series enters training. The model learns from a mixture of generative *mechanisms*, sampled
fresh for every training example. The code lives in
[`src/tsfm/data/`](../tsfm/src/tsfm/data/).

## Vocabulary

| object | shape | description |
|---|---|---|
| record | `[T, C]` | aligned multivariate array produced by a generator, with metadata |
| episode | `[C, L + H]` | what the network sees: `L` history steps, `H` horizon steps, channel roles and masks |
| batch | `[B, C, L_max + H_max]` | episodes aligned on the forecast origin, left-padded history, right-padded horizon |

Channel roles: `TARGET` (to forecast), `PAST_ONLY` covariates (visible in the history only) and
`KNOWN_FUTURE` covariates (calendar, planned events: visible over the horizon too).

## How one episode is made

`SyntheticEpisodeSource(config, split).episode(sample_id)` is a pure function of
`(seed, split, sample_id, config)`. Each sample owns independent random streams, so changing the
episode sampler does not change the underlying record.

```
univariate or multivariate   <- curriculum probability
generator family             <- generator_mix weights
L, H                         <- context_lengths, horizons
record                       <- generator(length = L + H + slack)
record                       <- augmentations (logged in metadata)
episode                      <- cut at a random origin, assign roles, apply missingness
```

## Generator families

| id | family | what it produces |
|---|---|---|
| G1 | `trend_seasonal` | trend plus up to 3 harmonic seasonalities with damping and noise (ForecastPFN-style) |
| G2 | `complex_periodic` | sine, square, sawtooth, triangle, asymmetric cycles, pulse trains, multi-period, chirps; frequency modulation; anti-aliased |
| G3 | `gaussian_process` | random kernel expressions over RBF, Matérn, Periodic, RationalQuadratic, Linear, White (KernelSynth-style); exact Cholesky up to 512 steps, random Fourier features beyond |
| G4 | `autoregressive` | stable AR / ARMA (drawn through partial autocorrelations), random walks, mean reversion, white noise |
| G5 | `regime_sde` | Markov-switching Ornstein–Uhlenbeck, drift switches, stochastic volatility, jump diffusion, regime GBM |
| G6 | `events_counts` | intermittent demand, count processes, spikes, bursts, level shifts, promotions |
| G7 | `composite` | non-linear combinations of the above: additive, amplitude-modulated, heteroscedastic, time-warped, then saturations and clipping |

Default mixture in [`configs/data_v1.yaml`](../tsfm/configs/data_v1.yaml): G1 15 %, G2 10 %, G3 15 %,
G4 15 %, G5 20 %, G6 10 %, G7 15 %. Time inside generators is expressed relative to the context
length, so periods and length-scales are sampled in units of what the model will actually see.

## Multivariate relations

The relational generator ([`multivariate.py`](../tsfm/src/tsfm/data/multivariate.py)) builds 2 to 4
channel records: independent covariates from the families above, then targets built from them with
one of these operators.

| code | operator | target depends on covariate `z` through |
|---|---|---|
| R1 | `signed_mix` | `a · z_t` |
| R2 | `lag` | `a · z_{t-τ}` |
| R3 | `nonlinear` | tanh, square, relu or threshold of `z_{t-τ}` |
| R4 | `shared_driver` | a latent driver both series observe with noise |
| R5 | `regime_response` | `a_{r_t} · z_{t-τ}`, two-state Markov regime |
| R6 | `event_effect` | a known event convolved with a carry-over kernel |
| R7 | `cointegration` | a shared random walk |
| R8 | decoys | covariates with no edge to any target |

Each target draws a covariate strength (`null`, `weak` or `strong`), so batches mix useful, weak
and useless relations. The current models are univariate and only read the targets; the
multivariate data is in place for a future cross-channel model.

## Masks and missing values

Four cases stay distinct:

1. an observed zero: value 0, observed
2. a missing value in the history: not observed, not supervised
3. the hidden future of a target: not observed, supervised (`loss_mask`)
4. padding: handled at collate time (`padding_mask`)

## Normalization and augmentations

Normalization is per channel, using the median and MAD of the **visible history only**. Future
labels are scaled with the past statistics for the loss and never contribute to them.

Augmentations respect each record's support: real-valued series can be scaled, shifted, sign-flipped,
noised or given outliers; non-negative series only get positive scales and offsets; integer count
series keep their count semantics. Operations apply jointly to all channels, so relations survive.

## Frozen validation suites

[`configs/holdout_v1.yaml`](../tsfm/configs/holdout_v1.yaml) declares named suites, each an overlay
of the training config with its own seed namespace. `scripts/build_holdouts.py` freezes them to
`data/holdouts/holdout_v1/<suite>/` as `episodes.npz` + `meta.jsonl` + `manifest.json`.
`FrozenEpisodeSet` refuses to load a suite whose files do not match the manifest hashes.

| suite | type | what is held out of training |
|---|---|---|
| `val_id`, `val_mv_id` | in distribution | nothing (new draws, univariate and multivariate) |
| `val_ood_noise` | parameter shift | the `extreme_noise` regime |
| `val_ood_period` | parameter shift | seasonal periods with P / L in [0.10, 0.14] |
| `val_ood_horizon` | parameter shift | horizon 384 (training max 192) |
| `val_ood_kinds` | unseen mechanism | `chirp` waveform (G2), `bursts` (G6) |
| `val_ood_relation` | unseen mechanism | cointegration (R7) |
| `test_syn_*` | final | built only with `--include-test`, for final reporting |

Everything held out for a suite is excluded from the training config, so out-of-distribution scores
are honest.

## Configs

| file | role |
|---|---|
| `data_v0.yaml` | univariate mix, no holdout exclusions (sanity and smoke runs) |
| `data_v1.yaml` | full training distribution: holdout exclusions, multivariate, curriculum |
| `holdout_v1.yaml` | frozen suites, as overlays of `data_v1.yaml` |
| `train_v0.yaml` | `data_v1` without multivariate episodes (first V0 / V1 runs) |
| `train_v1.yaml` | `train_v0` + short histories (32 and 64 steps), the current V1 recipe |
| `train_v1_aug.yaml` | `train_v1` with TiRex-style augmentations (ablation) |
| `train_v2.yaml` | `train_v1` + contexts up to 2048 and horizons up to 1024, for V2 |

Configs can `extend` another file and override part of it.

## Inspecting the data

```bash
uv run python scripts/inspect_samples.py --config configs/data_v1.yaml   # sample grids -> reports/data/samples/
uv run python scripts/audit_dataset.py --n 5000                           # statistics and baselines per kind
uv run python scripts/profile_generation.py --config configs/data_v1.yaml
uv run python scripts/evaluate_baselines.py                               # reference scores on the frozen suites
```

```python
from tsfm.data import SyntheticEpisodeSource, collate_episodes, load_config
from tsfm.data.frozen import FrozenEpisodeSet

src = SyntheticEpisodeSource(load_config("configs/data_v1.yaml"), split="train")
batch = collate_episodes([src.episode(i) for i in range(64)], pad_to_multiple=32)
# batch["target_future"] is a label: give it to the loss, never to model.forward

val = FrozenEpisodeSet("data/holdouts/holdout_v1/val_id")   # refuses to load if hashes differ
```
