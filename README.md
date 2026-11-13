# tsfm

A small (< 30M parameters) time-series foundation model trained **only on synthetic data**,
evaluated zero-shot on [GIFT-Eval](https://huggingface.co/datasets/Salesforce/GiftEval) (97 configurations).

- `tsfm/`: the Python package (data pipeline, model, training, benchmark) — see [`tsfm/README.md`](tsfm/README.md)
- `TSFM_DATA_REFERENCE.md`: specification of the synthetic data pipeline
- `roadmap.md`: project roadmap

## Approach

1. **Synthetic data, generated on the fly.** 7 generator families (trend + seasonality, complex
   periodic, Gaussian processes, autoregressive, regime-switching SDEs, events / counts,
   composites), random missingness and augmentations. Every episode is a pure function of its
   sample id: infinite, never-repeated data, exactly reproducible and resumable.
2. **Frozen validation suites** (in distribution + out of distribution: noise, periods, horizon,
   unseen waveforms) with sha256 manifests, to track progress without touching real data.
3. **Patch Transformer** with quantile outputs, trained with the pinball loss.
4. **Zero-shot evaluation on GIFT-Eval**, scored like the official leaderboard.

## Model V1 — architecture

| component | V1 |
|---|---|
| input | univariate target, history up to 1024 points, missing values masked |
| normalization | per series, median / MAD of the observed history (reversible) |
| tokenization | patches of 32 points, values ++ observation mask (64 features) |
| embedding | residual MLP 64 -> 384 -> 192 |
| backbone | 8 pre-norm blocks: RMSNorm, multi-head attention (3 heads, RoPE, bidirectional), SwiGLU (768) |
| head | last history token -> RMSNorm -> residual MLP 192 -> 768 -> 192 x 9 |
| output | 192 steps x 9 quantiles (0.1 ... 0.9) in one pass; longer horizons by feeding back the median |
| inference | symmetric averaging: forecast `x` and `-x`, average the reflected quantiles |
| parameters | **6.6M** (d_model 192) |
| loss | pinball loss over the 9 quantiles, in normalized space |
| training | AdamW (lr 1e-3, cosine, warmup 500), batch 64, synthetic data only |

## Results — GIFT-Eval (zero-shot, 97 configs)

Geometric mean over configs of MASE relative to Seasonal Naive (**lower is better**, < 1 beats
Seasonal Naive). WQL = mean weighted quantile loss (CRPS) relative to Seasonal Naive.

| model | training data | all | short | medium | long | WQL |
|---|---|---:|---:|---:|---:|---:|
| V0 (point forecast, MAE) | 0.5M synthetic episodes | 1.103 | – | – | – | – |
| V1 | 0.5M synthetic episodes | 1.096 | 0.977 | 1.252 | 1.295 | 0.746 |
| V1 + short contexts | 4M synthetic episodes (CPU, ~5 h) | 1.011 | 0.900 | 1.139 | 1.214 | 0.682 |
| V1 + short contexts | 30M synthetic episodes (GPU, ~6.6 h) | 0.966 | 0.863 | 1.086 | 1.156 | 0.660 |
| **V1 + symmetric averaging** | same checkpoint | **0.927** | **0.836** | **1.032** | **1.092** | **0.651** |
| *Seasonal Naive (official)* | – | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| *Naive (official)* | – | 1.270 | 1.143 | 1.463 | 1.453 | – |
| *chronos_bolt_small (official)* | real + synthetic data | 0.822 | 0.742 | 0.920 | 0.962 | 0.577 |
| *Timestral-5M (official)* | – | 0.705 | 0.673 | 0.737 | 0.759 | 0.479 |

V1 beats Seasonal Naive overall with synthetic data only. Its weak points are known: horizons
beyond 192 steps (produced by feeding back the median) and high-frequency series. Details,
ablations and throughput: [`tsfm/approaches/approach1_on_the_fly/README.md`](tsfm/approaches/approach1_on_the_fly/README.md).

## Quick start

```bash
cd tsfm
uv sync
uv run pytest -q
uv run python scripts/train.py --model v1 --data-config configs/train_v1.yaml --run-dir runs/approach1/v1_long
uv run python scripts/eval_gift.py --download
uv run python scripts/eval_gift.py --model runs/approach1/v1_long/best.pt --out-dir reports/gift_eval/approach1
```
