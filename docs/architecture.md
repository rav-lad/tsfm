# Architecture

All models are one class, `PatchTransformerV0` in
[`src/tsfm/models/patch_v0.py`](../tsfm/src/tsfm/models/patch_v0.py), configured by
`PatchV0Config`. Three presets are selectable with `--model` on the training CLI:

| preset | output | horizon in one pass | loss |
|---|---|---|---|
| `v0` | point forecast | 192 | MAE |
| `v1` | 9 quantiles, 0.1 … 0.9 | 192 | pinball |
| `v2` | 9 quantiles, one token per future patch | up to 1024 | pinball |

V1 is the current reference model (6.6M parameters).

<p align="center"><img src="assets/architecture.svg" alt="Animated architecture diagram of V1" width="100%"></p>

## Forward pass

The model is univariate: it reads the `TARGET` channels of a batch and ignores covariates.
Shapes below use `N` series, `L` history steps, `P = L / 32` patches and `d = 192`.

| step | operation | shape |
|---|---|---|
| input | history ending at the forecast origin, missing values masked | `[N, L]` |
| normalization | per series, median / MAD of the observed history (done in `collate_episodes`, reversible) | `[N, L]` |
| left padding | pad to a multiple of 32 so the last patch ends exactly at the origin | `[N, 32P]` |
| tokenization | each patch = 32 values ++ 32 observation bits (missing values set to 0) | `[N, P, 64]` |
| embedding | residual MLP 64 → 384 → 192 | `[N, P, d]` |
| backbone | 8 pre-norm blocks: RMSNorm, multi-head attention (3 heads, RoPE, bidirectional), RMSNorm, SwiGLU (hidden 768) | `[N, P, d]` |
| head (V0, V1) | last history token → RMSNorm → residual MLP 192 → 768 → 192 × Q | `[N, 192, Q]` |
| head (V2) | each future token → RMSNorm → residual MLP → 32 × Q | `[N, 32K, Q]` |

`Q = 1` for V0 and `Q = 9` for V1 and V2.

**Positions.** RoPE positions are counted backwards from the origin (the last history patch is at
position 0), so left padding never shifts the real patches. Padded patches are masked out of the
attention keys.

**Attention is bidirectional.** The whole history is known at forecast time, so every history patch
attends to every other one. Causality is enforced at the episode level instead: the model only reads
`input_values[..., :origin]`.

**Quantile crossing.** At inference (`model.eval()`) the 9 quantiles are sorted, so they never cross.

## V2: future patch tokens

V2 appends `K = ceil(horizon / 32)` copies of one learned embedding after the history tokens. They sit
at positions `1 … K` after the origin, attend to everything, and each one is decoded into its own
32-step block of quantiles. The horizon is produced in one pass up to `max_horizon = 1024`, with no
rollout. This follows the decoupled-horizon idea of Timestral and Chronos-2. V2 is trained with
[`configs/train_v2.yaml`](../tsfm/configs/train_v2.yaml) (contexts up to 2048, horizons up to 1024).

## Horizons beyond `max_horizon`: rollout

When the requested horizon is longer than one pass (more than 192 steps for V0 and V1), the model
appends its point forecast (the median for a quantile model) to the history as observed values and
runs again. Quantiles after the first 192 steps are then conditional on that median path, so
uncertainty is underestimated. This is the main cause of the weaker medium and long GIFT-Eval scores.

## Symmetric averaging at inference

The benchmark predictor ([`benchmarks/predictor.py`](../tsfm/src/tsfm/benchmarks/predictor.py))
forecasts both `x` and `-x` in the same batch and averages the reflected forecast:

```
q_tau = ( q_tau(x) - q_{1-tau}(-x) ) / 2
```

The quantile levels are symmetric around 0.5, so the median stays the middle quantile. It costs
nothing at training time and doubles inference cost. On the 30M-episode checkpoint it moves GIFT-Eval
relative MASE from 0.966 to 0.927. Disable it with `--no-symmetric` in `eval_gift.py`.

## Leakage guards

- `collate_episodes` computes normalization statistics on the visible history only.
- `split_batch` separates the labels (`target_future`, `loss_mask`) from the model inputs.
- `model.forward` raises if it receives a label tensor.
- Every non-observed position in `input_values` is zeroed before the model sees it.

## Configuration reference

| field | V1 value | meaning |
|---|---:|---|
| `patch_size` | 32 | steps per token |
| `d_model` | 192 | width of the residual stream |
| `n_layers` | 8 | Transformer blocks |
| `n_heads` | 3 | attention heads (head dim 64) |
| `ffn_hidden` | 768 | SwiGLU hidden width |
| `embed_hidden` | 384 | hidden width of the patch embedding MLP |
| `head_hidden` | 768 | hidden width of the output MLP |
| `max_horizon` | 192 | steps produced in one pass (1024 for V2) |
| `rope_base` | 10000 | RoPE frequency base |
| `quantiles` | 0.1 … 0.9 | empty tuple = point forecast |
| `future_tokens` | false | true for V2 |

These sizes match Timestral-5M (d_model 192, 8 blocks, patch 32), which makes that model a direct
reference for what the same architecture reaches with real data in pre-training.
