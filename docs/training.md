# Training

Training reads an infinite stream of synthetic episodes generated on the fly by DataLoader workers.
Nothing is stored on disk except the frozen validation suites and the checkpoints.

## Run a training

From `tsfm/`:

```bash
uv run python scripts/build_holdouts.py          # once: frozen validation suites -> data/holdouts/holdout_v1/

uv run python scripts/train.py --model v1 --data-config configs/train_v1.yaml \
    --run-dir runs/v1 --total-episodes 4000000
```

A run directory contains:

| file | content |
|---|---|
| `metrics.jsonl` | loss, learning rate, gradient norm, episodes/s, fraction of time waiting for data |
| `last.pt` | latest checkpoint (model, optimizer, step, configs) |
| `best.pt` | best checkpoint on `best_metric_suite` (default `val_id`) |
| `eval.json` | scores on every `val_*` suite at the end of training |
| `crash.pt` | written only if 20 consecutive steps produce non-finite gradients |

Resume with `--resume`. End a session early with `--stop-after-hours 11` or `--stop-after-steps N`
(useful on time-limited GPUs), then rerun the same command with `--resume`.

## Determinism and resume

Batch `k` always holds sample ids `[k·B, (k+1)·B)`. Worker `w` of `n` builds batches `w, w+n, w+2n…`
and the DataLoader yields them round-robin, so the stream is identical for any number of workers.
Resuming is `start_batch = steps_done`: the run continues on exactly the data it would have seen.
Workers build whole batches (generate and collate) so only a few numpy arrays cross process
boundaries.

```python
from tsfm.training import make_frozen_loader, make_train_loader, split_batch, to_torch

loader = make_train_loader("configs/train_v1.yaml", batch_size=64, num_workers=8, start_batch=steps_done)
for batch in loader:
    inputs, labels = split_batch(to_torch(batch, device))   # labels: target_future, loss_mask
    out = model(**inputs)                                    # the model never sees the labels

val = make_frozen_loader("data/holdouts/holdout_v1/val_id", batch_size=64)
```

## Optimization

| setting | default |
|---|---|
| loss | MAE (V0) or pinball over 9 quantiles (V1, V2), in normalized space, on supervised positions only |
| optimizer | AdamW, weight decay 0.1 (none on norms and biases) |
| learning rate | 1e-3, 500 warmup steps, cosine decay to 10 % |
| batch size | 64 episodes |
| gradient clipping | 1.0 |
| mixed precision | autocast on CUDA: bf16 when supported, else fp16 with loss scaling; `--no-amp` for fp32 |
| periodic eval | every 1000 steps on the first 20 batches of `val_id` |

Optional loss weightings (`--loss-weighting harmonic naive_scale`): `harmonic` weights horizon step
`h` by `1/h` (Timer-M1); `naive_scale` divides by the one-step variability of the history (MASE-like).
Neither improved GIFT-Eval in the ablations (see [evaluation.md](evaluation.md#ablations)).

Every field of `TrainConfig` in [`trainer.py`](../tsfm/src/tsfm/training/trainer.py) is a CLI flag:

```bash
uv run python scripts/train.py --help
```

## Hardware

**CPU or CUDA.** `uv sync` installs PyTorch for the platform. With 16 threads, on-the-fly
generation produces about 1 100 episodes/s; CPU training is model-bound at about 225 episodes/s.

**AMD GPU on Windows (ROCm).** Create a separate environment once, then use its interpreter:

```powershell
powershell -ExecutionPolicy Bypass -File scripts/setup_rocm_windows.ps1    # creates .venv-rocm
```

On an RX 6800 XT, fp32 is faster than bf16 (about 2 060 vs 1 490 episodes/s) and better at level
anchoring, so the launch scripts use `--no-amp`.

**Kaggle.** [`approaches/approach1_on_the_fly/train_kaggle.ipynb`](../tsfm/approaches/approach1_on_the_fly/train_kaggle.ipynb)
runs the same command in time-limited sessions and resumes from the previous session's output.
Kaggle's 4 vCPUs cannot keep a T4 fed (75 % of the time is spent waiting for data), so a local
multi-core CPU is faster.

## Launch scripts

Run from `tsfm/`; each one is safe to rerun and resumes where it stopped.

| script | what it does |
|---|---|
| `approaches/approach1_on_the_fly/run_v1_gpu_8h.sh` / `.ps1` | V1, 30M episodes on the local GPU (~7 h), then GIFT-Eval |
| `approaches/approach1_on_the_fly/run_ablations_p1.sh` | phase-1 ablations (loss weightings, augmentations), 4M episodes each |
| `approaches/approach1_on_the_fly/run_v2.sh` | V2 (future tokens, horizons to 1024), 4M episodes, then GIFT-Eval with 2048-step contexts |

## Scripts reference

| script | purpose |
|---|---|
| `scripts/train.py` | train a model on the synthetic stream |
| `scripts/eval_gift.py` | download GIFT-Eval and evaluate a checkpoint or a baseline zero-shot |
| `scripts/build_holdouts.py` | freeze the validation (and optionally test) suites |
| `scripts/evaluate_baselines.py` | reference scores of simple baselines on the frozen suites |
| `scripts/inspect_samples.py` | plot sample grids per generator |
| `scripts/audit_dataset.py` | statistics and baseline scores per generator kind |
| `scripts/profile_generation.py` | generation speed per generator |
| `scripts/profile_loader.py` | DataLoader throughput for several worker counts |
| `scripts/setup_rocm_windows.ps1` | ROCm PyTorch environment for AMD GPUs on Windows |
