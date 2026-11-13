# Approach 1 — on-the-fly generation

Every training episode is generated when the DataLoader asks for it
(`SyntheticEpisodeSource.episode(sample_id)`): infinite data, never repeated,
deterministic per sample id, so `--resume` replays the exact stream.

```bash
uv run python scripts/train.py --model v1 --data-config configs/train_v1.yaml \
    --run-dir runs/approach1/v1_long --total-episodes 4000000
uv run python scripts/eval_gift.py --model runs/approach1/v1_long/best.pt --out-dir reports/gift_eval/approach1
```

Local GPU (AMD RX 6800 XT, ROCm on Windows): `powershell -ExecutionPolicy Bypass -File scripts/setup_rocm_windows.ps1`
once, then `powershell -ExecutionPolicy Bypass -File approaches/approach1_on_the_fly/run_v1_gpu_8h.ps1` (V1, 30M episodes, bf16, ~7.5 h, then GIFT-Eval;
rerun to resume).

Kaggle: `train_kaggle.ipynb` (same command, `--stop-after-hours` sessions, resume from the previous output).

## Runs (`runs/approach1/`, reports in `reports/gift_eval/approach1/`)

| run | model | data config | episodes | wall time (CPU) | GIFT rel. MASE (short / medium / long) | rel. WQL | val_id MASE |
|---|---|---|---:|---:|---|---:|---:|
| `v0` | V0, point (MAE) | `train_v0.yaml` | 0.5M | – | 1.103 | – | 1.526 |
| `v1` | V1, 9 quantiles | `train_v0.yaml` | 0.5M | ~35 min | 1.096 (0.977 / 1.252 / 1.295) | 0.746 | 1.576 |
| `v1_long` | V1, 9 quantiles | `train_v1.yaml` (+32/64 contexts) | 4M | ~5 h | 1.011 (0.900 / 1.139 / 1.214) | 0.682 | 1.369 |
| `v1_gpu_8h` | V1, 9 quantiles | `train_v1.yaml` | 30M | ~6.6 h (GPU) | 0.966 (0.863 / 1.086 / 1.156) | 0.660 | 1.331 |
| `v1_gpu_8h` + symmetric averaging | same checkpoint | – | – | – | **0.927** (0.836 / 1.032 / 1.092) | **0.651** | – |

Symmetric averaging (`eval_gift.py` default since 2026-11-13, reports suffixed `_sym`): the predictor
also forecasts `-x` in the same batch and averages the reflected forecast, `q_tau = (q_tau(x) - q_{1-tau}(-x)) / 2`
(Timestral, TempoPFN). Free at training time, 2x inference cost; `--no-symmetric` for a single pass.
Rows above `v1_gpu_8h + symmetric` are single-pass.

### Phase-1 ablations (2026-11-13, `run_ablations_p1.sh`)

v1_long recipe (V1, `train_v1.yaml`, 4M episodes, batch 64), local GPU fp32, ~35-45 min each.
All scores with symmetric averaging. "RW 1-step" = MAE / naive MAE at h=1 on driftless random
walks (level anchoring; 1.00 = perfect).

| run | change | GIFT rel. MASE (short / medium / long) | RW 1-step |
|---|---|---|---:|
| `v1_long` (baseline) | – | **0.963** (0.874 / 1.068 / 1.117) | 1.13 |
| `abl_harmonic` | loss weights 1/h (Timer-M1) | 0.971 (**0.852** / 1.125 / 1.182) | **1.06** |
| `abl_naive_scale` | loss weights 1 / mean abs step of the history (MASE-like) | 1.029 (0.933 / 1.146 / 1.193) | 1.11 |
| `abl_aug` | no scale/offset aug, + TiRex AM / censor / spikes | 1.023 (0.912 / 1.176 / 1.203) | – |

None beats the baseline. 1/h fixes level anchoring and short horizons but starves the far
horizon (worse medium / long, amplified by the rollout); naive-scale weighting over-weights
smooth series; the TiRex augmentations hurt at this budget. Kept: fp32 (bf16 was slower on the
RX 6800 XT and worse at level anchoring).

References (official GIFT-Eval): Seasonal Naive 1.000, chronos_bolt_small 0.822 / WQL 0.577,
Timestral-5M 0.705 / 0.479 (same architecture size: d_model 192, 8 blocks, patch 32).

`v1_long` notes:
- 80 / 97 configs improve on `v1`; beats Seasonal Naive on 61 / 97 (v1: 48).
- m4_yearly 1.83 -> 0.96 (short contexts); m4_daily unchanged (1.51).
- Worst: high frequencies with medium / long horizons (10S 2.41, 5T 1.20, 10T 1.14 vs Seasonal Naive),
  i.e. horizons > 192 obtained by rollout; bitbrains / solar ~2x worse than Chronos.
- Coverage of [q0.1, q0.9]: 80 % in distribution, 69 % on `val_ood_horizon` (384, rollout).
- val_id (20-batch periodic eval) was still decreasing at the end: under-trained.

Inference context length (`v1_long`, `--max-context`): 512 -> 1.021, **1024 -> 1.011**, 2048 -> 1.012;
best context chosen per config (oracle) 0.989. Nothing to gain at inference.

## Throughput

| setup | generation | training |
|---|---:|---:|
| local, 8 workers (16 threads), CPU training | ~1 080 ep/s | ~225 ep/s (model-bound) |
| local, RX 6800 XT (ROCm, Windows), 10 workers, batch 64 | ~1 200-1 450 ep/s | ~1 200-1 450 ep/s, data wait 1 % |
| Kaggle, T4 + 4 vCPU (2 physical cores) | 118 ep/s (2 workers), 81 (4 workers) | ~85 ep/s, **data wait 75 %** |

On Kaggle the T4 waits for the CPU three quarters of the time: slower than the local CPU.
Locally the 16 threads keep the RX 6800 XT fed, so on-the-fly generation is enough.
