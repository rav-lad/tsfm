<h1 align="center">tsfm</h1>

<p align="center">
  <b>A 6.6M-parameter time-series foundation model trained only on synthetic data.</b><br>
  It forecasts real series zero-shot, and beats Seasonal Naive on the
  <a href="https://huggingface.co/datasets/Salesforce/GiftEval">GIFT-Eval</a> benchmark without seeing a single real data point during training.
</p>

<p align="center">
  <img src="https://img.shields.io/badge/python-3.13-3776AB?logo=python&logoColor=white" alt="Python 3.13">
  <img src="https://img.shields.io/badge/PyTorch-2.x-EE4C2C?logo=pytorch&logoColor=white" alt="PyTorch">
  <img src="https://img.shields.io/badge/GIFT--Eval%20rel.%20MASE-0.927-245A9C" alt="GIFT-Eval relative MASE 0.927">
  <img src="https://img.shields.io/badge/training%20data-100%25%20synthetic-C25E14" alt="100% synthetic training data">
  <a href="LICENSE"><img src="https://img.shields.io/badge/license-MIT-5A6576" alt="MIT license"></a>
</p>

<p align="center">
  <img src="docs/assets/architecture.svg" alt="Animated architecture of PatchTransformer V1: a history is normalized and cut into 32-step patches, embedded, passed through 8 Transformer blocks, and the last patch token is decoded into 192 steps by 9 quantiles." width="100%">
</p>

## Why

Most forecasting foundation models are pre-trained on large collections of real series, which makes
it hard to know what they generalize from and whether the benchmark leaked into training.
This project asks a narrower question: **how far can a small model (< 30M parameters) generalize
when it is trained only on a compositional distribution of synthetic processes?**

## Highlights

- **Synthetic data generated on the fly.** Seven generator families (trends and seasonality,
  complex periodicity, Gaussian processes, autoregressive, regime-switching SDEs, events and counts,
  non-linear composites) plus a multivariate relational generator. Every episode is a pure function
  of its sample id, so the stream is infinite, never repeats, and resumes exactly.
- **No leakage by construction.** Normalization uses the visible history only, labels are split off
  before `model.forward`, and the model refuses to receive them.
- **Frozen synthetic validation suites** (in-distribution and out-of-distribution) with sha256
  manifests, to measure progress without touching real data.
- **Patch Transformer with quantile outputs**, trained with the pinball loss, 192 steps × 9 quantiles
  in one forward pass.
- **GIFT-Eval re-implemented without gluonts**, validated against the official Seasonal Naive scores
  on all 97 configurations.

## Results

Zero-shot on GIFT-Eval (97 configurations). Geometric mean of MASE relative to Seasonal Naive:
lower is better, below 1.0 beats Seasonal Naive. WQL is the mean weighted quantile loss (CRPS),
also relative to Seasonal Naive.

| model | training data | all | short | medium | long | WQL |
|---|---|---:|---:|---:|---:|---:|
| V0 (point forecast) | 0.5M synthetic episodes | 1.103 | – | – | – | – |
| V1 | 0.5M synthetic episodes | 1.096 | 0.977 | 1.252 | 1.295 | 0.746 |
| V1 + short contexts | 4M synthetic episodes | 1.011 | 0.900 | 1.139 | 1.214 | 0.682 |
| V1 + short contexts | 30M synthetic episodes | 0.966 | 0.863 | 1.086 | 1.156 | 0.660 |
| **V1 + symmetric averaging** | same checkpoint | **0.927** | **0.836** | **1.032** | **1.092** | **0.651** |
| *Seasonal Naive* | – | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| *Naive* | – | 1.270 | 1.143 | 1.463 | 1.453 | – |
| *Chronos-Bolt small* | real + synthetic | 0.822 | 0.742 | 0.920 | 0.962 | 0.577 |
| *Timestral-5M* | real + synthetic | 0.705 | 0.673 | 0.737 | 0.759 | 0.479 |

Rows in italics are official leaderboard scores. The main known weaknesses are horizons beyond 192
steps (produced by feeding the median back) and high-frequency series. Full numbers, ablations and
throughput are in [docs/evaluation.md](docs/evaluation.md).

## Quick start

Requires [uv](https://docs.astral.sh/uv/) and Python 3.13. All commands run from `tsfm/`.

```bash
cd tsfm
uv sync                                   # create the environment
uv run pytest -q                          # run the test suite

# build the frozen validation suites (once)
uv run python scripts/build_holdouts.py

# train V1 (9 quantiles) on the on-the-fly synthetic stream
uv run python scripts/train.py --model v1 --data-config configs/train_v1.yaml --run-dir runs/v1

# download GIFT-Eval (1.6 GB) and evaluate zero-shot
uv run python scripts/eval_gift.py --download
uv run python scripts/eval_gift.py --model runs/v1/best.pt
```

Forecast from Python:

```python
import numpy as np
from tsfm.benchmarks.predictor import model_predictor
from tsfm.training.trainer import load_model

predict = model_predictor(load_model("runs/v1/best.pt"), device="cpu")

history = np.sin(np.arange(500) / 8) + 0.01 * np.arange(500)   # raw values, NaN = missing
median, quantiles = predict([history], prediction_length=96, seasonality=1)
# median: [1, 96]   quantiles: [1, 96, 9] at levels 0.1 ... 0.9, in the original scale
```

See [docs/training.md](docs/training.md) for GPU setup (CUDA, ROCm on Windows, Kaggle) and every
training flag.

## Documentation

| page | contents |
|---|---|
| [Architecture](docs/architecture.md) | model V0 / V1 / V2, tensor shapes, normalization, rollout, symmetric averaging |
| [Synthetic data](docs/data.md) | episode contract, generator families, multivariate relations, masks, holdout suites |
| [Training](docs/training.md) | data loader, determinism and resume, losses, configs, hardware, scripts |
| [Evaluation](docs/evaluation.md) | GIFT-Eval protocol, full results, ablations, throughput |

## Repository layout

```
tsfm/                      Python package and experiments (run commands from here)
├── src/tsfm/
│   ├── data/              synthetic pipeline: generators, episodes, masks, collate, holdouts
│   ├── models/            patch Transformer (V0, V1, V2 presets)
│   ├── training/          loader, losses, trainer
│   └── benchmarks/        GIFT-Eval re-implementation and model predictor
├── configs/               data, holdout and training distributions (YAML)
├── scripts/               train, evaluate, build holdouts, profile, inspect
├── approaches/            launch scripts for training runs (local GPU, Kaggle notebook)
├── benchmarks/gift_eval/  official reference scores used for comparison
└── tests/
docs/                      documentation and the architecture animation
```

Generated folders (`data/`, `runs/`, `reports/`) are git-ignored and rebuilt by the scripts.

## References

The design borrows from, and is compared against:
Timestral (Mistral AI, 2026), Timer-M1 (Tsinghua, 2026), t0 (The Forecasting Company, 2026),
Chronos and Chronos-Bolt (Amazon, 2024), TempoPFN, TiRex, ForecastPFN and KernelSynth.
Benchmark: Aksu et al., *GIFT-Eval: A Benchmark for General Time Series Forecasting Model Evaluation*, 2024.

## License

[MIT](LICENSE). The GIFT-Eval reference files in `tsfm/benchmarks/gift_eval/reference/` are
Apache-2.0, Copyright (c) 2024 Salesforce, Inc. (see the `NOTICE` file there).
