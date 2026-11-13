# Evaluation

## GIFT-Eval protocol

[`src/tsfm/benchmarks/gift_eval.py`](../tsfm/src/tsfm/benchmarks/gift_eval.py) re-implements the
official [GIFT-Eval](https://github.com/SalesforceAIResearch/gift-eval) harness without gluonts
(which pins numpy 1.26). It reproduces the official Seasonal Naive scores on all 97 configurations
(MASE within 1e-7, ND within 0.1 %).

- **97 configurations**: 55 short-term datasets, 21 of which also have medium and long terms.
- **Prediction length**: the base length of the frequency × 1, 10 or 15 for short, medium and long.
- **Test windows**: the last `windows × prediction_length` points, in non-overlapping rolling windows.
- **Multivariate datasets** are split into univariate series.
- **MASE** is relative to the in-sample seasonal error; **WQL** (the CRPS column of the leaderboard)
  is the mean over quantiles 0.1 … 0.9 of the weighted pinball loss.
- **Aggregate**: geometric mean over configurations of the score divided by Seasonal Naive's.
  Below 1.0 beats Seasonal Naive.

```bash
uv run python scripts/eval_gift.py --download                       # 1.6 GB -> data/gift_eval/
uv run python scripts/eval_gift.py --model runs/v1/best.pt          # checkpoint, zero-shot
uv run python scripts/eval_gift.py --model seasonal_naive           # or a baseline
uv run python scripts/eval_gift.py --model runs/v1/best.pt --terms short --datasets m4
```

Useful flags: `--max-context` (default 1024), `--no-symmetric` (single pass),
`--out-dir` (default `reports/gift_eval/`). The official reference scores used for comparison live in
[`tsfm/benchmarks/gift_eval/reference/`](../tsfm/benchmarks/gift_eval/reference/).

## Results

| run | model | data config | episodes | wall time | rel. MASE (short / medium / long) | rel. WQL | val_id MASE |
|---|---|---|---:|---:|---|---:|---:|
| `v0` | V0, point | `train_v0.yaml` | 0.5M | – | 1.103 | – | 1.526 |
| `v1` | V1 | `train_v0.yaml` | 0.5M | ~35 min CPU | 1.096 (0.977 / 1.252 / 1.295) | 0.746 | 1.576 |
| `v1_long` | V1 | `train_v1.yaml` | 4M | ~5 h CPU | 1.011 (0.900 / 1.139 / 1.214) | 0.682 | 1.369 |
| `v1_gpu_8h` | V1 | `train_v1.yaml` | 30M | ~6.6 h GPU | 0.966 (0.863 / 1.086 / 1.156) | 0.660 | 1.331 |
| `v1_gpu_8h` + symmetric | same checkpoint | – | – | – | **0.927** (0.836 / 1.032 / 1.092) | **0.651** | – |

All rows except the last are single-pass. Official references: Seasonal Naive 1.000,
Chronos-Bolt small 0.822 (WQL 0.577), Timestral-5M 0.705 (WQL 0.479). Timestral-5M has the same
architecture size, so the gap measures what real data and a larger training budget add.

### What the numbers say

- Short contexts in training (`train_v1.yaml`) improve 80 of 97 configurations; `m4_yearly` goes from
  1.83 to 0.96. The model beats Seasonal Naive on 61 of 97 configurations (48 before).
- The weakest cases are high frequencies with medium and long horizons (10-second data 2.41,
  5-minute 1.20, 10-minute 1.14), where horizons beyond 192 steps come from the rollout.
  Bitbrains and solar are about twice as bad as Chronos.
- Coverage of the [q0.1, q0.9] interval: 80 % in distribution, 69 % at horizon 384 (rollout).
- `val_id` was still decreasing at the end of every run: the model is under-trained.
- Inference context length does not matter: 512 → 1.021, 1024 → 1.011, 2048 → 1.012 on `v1_long`.
  Choosing the best context per configuration (an oracle) only reaches 0.989.

## Ablations

Phase 1 (`run_ablations_p1.sh`): `v1_long` recipe (V1, `train_v1.yaml`, 4M episodes, batch 64), local
GPU in fp32, about 40 minutes each. All scores use symmetric averaging. "RW 1-step" is the MAE at
`h = 1` on driftless random walks divided by the naive MAE (1.00 is perfect level anchoring).

| run | change | rel. MASE (short / medium / long) | RW 1-step |
|---|---|---|---:|
| `v1_long` | baseline | **0.963** (0.874 / 1.068 / 1.117) | 1.13 |
| `abl_harmonic` | loss weight `1/h` (Timer-M1) | 0.971 (**0.852** / 1.125 / 1.182) | **1.06** |
| `abl_naive_scale` | loss weight 1 / mean absolute step of the history | 1.029 (0.933 / 1.146 / 1.193) | 1.11 |
| `abl_aug` | no scale / offset, + TiRex amplitude modulation, censoring, spikes | 1.023 (0.912 / 1.176 / 1.203) | – |

None beats the baseline. `1/h` fixes level anchoring and short horizons but starves the far horizon,
which the rollout then amplifies. Naive-scale weighting over-weights smooth series. The TiRex
augmentations hurt at this budget.

## Throughput

| setup | generation | training |
|---|---:|---:|
| local CPU, 8 workers (16 threads) | ~1 080 episodes/s | ~225 episodes/s (model-bound) |
| RX 6800 XT (ROCm, Windows), 10 workers, batch 64 | ~1 200–1 450 episodes/s | ~1 200–1 450 episodes/s, 1 % data wait |
| Kaggle T4 + 4 vCPU | 81–118 episodes/s | ~85 episodes/s, 75 % data wait |

On a multi-core machine, on-the-fly generation keeps the GPU fed, so nothing needs to be
pre-generated.

## Next steps

- V2 (future patch tokens, horizons up to 1024 in one pass) to remove the rollout.
- Longer training: validation loss had not plateaued at 30M episodes.
- A cross-channel model that uses the multivariate data already generated.
