# tsfm — synthetic data pipeline and model

Synthetic-only data for a <30M-parameter time-series foundation model.
Specification: `../TSFM_DATA_REFERENCE.md` · roadmap: `../roadmap.md`.

**Status:** data sprints D1–D5 are done; model V1 (6.6M parameters, 9 quantiles) reaches
GIFT-Eval rel. MASE **0.927** zero-shot (architecture and results: [`../README.md`](../README.md)).

- D1: episode contract, deterministic per-sample RNG, masks, leak-free normalization, collate
- D2/D3: the 7 generator families G1–G7 (`src/tsfm/data/generators/`)
- D4: named, frozen evaluation suites (H1 / H2 / H3), sha256 manifests, baseline reference scores
- D5: multivariate relational generator (R1–R8), TARGET / PAST_ONLY / KNOWN_FUTURE roles,
  decoys, multivariate curriculum

```bash
uv sync                                                     # CPU (or CUDA on Linux) environment
powershell -ExecutionPolicy Bypass -File scripts/setup_rocm_windows.ps1   # AMD GPU on Windows -> .venv-rocm
uv run pytest -q                                            # 156 tests
uv run python scripts/inspect_samples.py --config configs/data_v1.yaml   # grids -> reports/data/samples/
uv run python scripts/profile_generation.py --config configs/data_v1.yaml
uv run python scripts/audit_dataset.py --n 5000             # stats + baselines per kind
uv run python scripts/build_holdouts.py                     # freeze val_* suites -> data/holdouts/holdout_v1/
uv run python scripts/profile_loader.py --workers 0 4 8     # DataLoader episodes/s
uv run python scripts/train.py --run-dir runs/approach1/v0     # model V0 -> <run>/{metrics.jsonl,last.pt,best.pt,eval.json}
uv run python scripts/train.py --model v1 --data-config configs/train_v1.yaml --run-dir runs/approach1/v1_long  # V1: 9 quantiles
uv run python scripts/eval_gift.py --download              # GIFT-Eval (1.6 GB) -> data/gift_eval/
uv run python scripts/eval_gift.py --model runs/approach1/v0/best.pt  # zero-shot, 97 configs -> reports/gift_eval/
uv run python scripts/evaluate_baselines.py                 # reference scores -> reports/data/baselines_holdout_v1.md
```

```python
from tsfm.data import SyntheticEpisodeSource, collate_episodes, load_config
from tsfm.data.frozen import FrozenEpisodeSet

src = SyntheticEpisodeSource(load_config("configs/data_v1.yaml"), split="train")
batch = collate_episodes([src.episode(i) for i in range(64)], pad_to_multiple=32)
# batch["target_future"] is a label: send it to the loss, never to model.forward

val = FrozenEpisodeSet("data/holdouts/holdout_v1/val_id")   # refuses to load if hashes differ
```

PyTorch side (`src/tsfm/training/`): workers generate and collate whole batches;
batch `k` always holds sample ids `[k*B, (k+1)*B)`, whatever `num_workers`.

```python
from tsfm.training import make_frozen_loader, make_train_loader, split_batch, to_torch

loader = make_train_loader("configs/data_v1.yaml", batch_size=64, num_workers=8,
                           start_batch=steps_done)                 # resume = skip batches
for batch in loader:
    inputs, labels = split_batch(to_torch(batch, device))         # labels: target_future, loss_mask
    pred = model(**inputs)                                         # never sees labels

val = make_frozen_loader("data/holdouts/holdout_v1/val_id", batch_size=64)
```

Training runs are grouped by approach in `approaches/` (README with method and results, Kaggle notebook):
- `approach1_on_the_fly/`: every episode generated when the DataLoader asks for it

Configs:
- `configs/data_v0.yaml`: univariate mix, no holdout exclusions (sanity / smoke runs)
- `configs/data_v1.yaml`: **training** distribution: holdout exclusions + multivariate + curriculum
- `configs/holdout_v1.yaml`: frozen suites as overlays of `data_v1.yaml`

| suite | type | what is held out of training |
|---|---|---|
| `val_id`, `val_mv_id` | H1 | nothing (new draws, uni / multivariate) |
| `val_ood_noise` | H2 | `extreme_noise` regime |
| `val_ood_period` | H2 | seasonal periods with P/L in [0.10, 0.14] |
| `val_ood_horizon` | H2 | horizon 384 (training max 192) |
| `val_ood_kinds` | H3 | `chirp` waveform (G2), `bursts` (G6) |
| `val_ood_relation` | H3 | cointegration (R7) |
| `test_syn_*` | final | built only with `--include-test`, for final reporting |

Layout: `src/tsfm/data/` (the reference calls it `src/data/`; it is namespaced
here so the package can be installed with `pip install` on Kaggle/Colab).
Sources are `[T, C]`, episodes and batches are `[C, L+H]`.
`data/` and `reports/` are generated and git-ignored.
