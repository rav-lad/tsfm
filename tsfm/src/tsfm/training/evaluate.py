"""Model scores on frozen suites, comparable to ``scripts/evaluate_baselines.py``.

Per target series: MAE on the leak-free normalized scale and MASE (MAE divided
by the in-sample one-step naive MAE of the forward-filled history), next to the
same baselines (naive, seasonal naive, drift, mean, best-of-four). Baselines
are deterministic, so they are computed once per suite and cached.

Quantile models (V1) also get, per series, the scaled quantile loss
``SQL = 2 * mean pinball / MASE scale`` (a CRPS estimate in MASE units; for a
point forecast it equals MASE) and the empirical coverage of the central
interval between the lowest and highest levels (0.1-0.9 -> nominal 80%).
"""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

from tsfm.data import Role
from tsfm.data.evaluation import _ffill, baseline_forecasts, mae, mase_scale

from .loader import FrozenBatchDataset, split_batch, to_torch

BASELINES = ("naive", "seasonal_naive", "drift", "mean")


def _rows(batch: dict[str, np.ndarray], ids: range) -> list[dict[str, Any]]:
    """One entry per supervised target series: history, labels, mask, MASE scale."""
    W = batch["input_values"].shape[-1]
    origin = W - batch["target_future"].shape[-1]
    rows = []
    for b, i in enumerate(ids):
        L, H = (int(v) for v in batch["lengths"][b])
        for c in np.flatnonzero(batch["channel_roles"][b] == Role.TARGET):
            m = batch["loss_mask"][b, c, :H]
            if not m.any():
                continue
            hist = _ffill(batch["input_values"][b, c, origin - L : origin].astype(float),
                          batch["input_mask"][b, c, origin - L : origin])
            rows.append({"episode": i, "b": b, "c": int(c), "H": H, "hist": hist,
                         "y": batch["target_future"][b, c, :H].astype(float), "mask": m,
                         "scale": mase_scale(hist)})
    return rows


def _baseline_scores(rows: list[dict[str, Any]]) -> dict[str, np.ndarray]:
    out = {k: [] for k in BASELINES}
    for r in rows:
        for k, pred in baseline_forecasts(r["hist"], r["H"]).items():
            out[k].append(mae(r["y"], pred, r["mask"]) / r["scale"])
    scores = {k: np.array(v) for k, v in out.items()}
    scores["best"] = np.min(np.stack([scores[k] for k in BASELINES]), axis=0)
    return scores


@torch.no_grad()
def evaluate_suite(
    model: nn.Module,
    suite_dir: str | Path,
    device: str | torch.device = "cpu",
    batch_size: int = 64,
    max_batches: int | None = None,
    cache: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Median MASE of the model and of the baselines on one frozen suite.

    ``cache`` (any dict, reused across calls) keeps the suite and its baseline
    scores so periodic evaluation only pays for the model forward passes.
    """
    key = f"{Path(suite_dir).resolve()}|{batch_size}|{max_batches}"
    cache = cache if cache is not None else {}
    if key not in cache:
        ds = FrozenBatchDataset(suite_dir, batch_size)
        n = len(ds) if max_batches is None else min(len(ds), max_batches)
        batches, rows = [], []
        for k in range(n):
            batch = ds[k]
            ids = range(k * batch_size, k * batch_size + batch["input_values"].shape[0])
            batches.append(batch)
            rows.append(_rows(batch, ids))
        families = [ds.suite[r["episode"]].meta["family"] for rs in rows for r in rs]
        cache[key] = {"batches": batches, "rows": rows, "families": np.array(families),
                      "baselines": _baseline_scores([r for rs in rows for r in rs])}
    entry = cache[key]

    was_training = model.training
    model.eval()
    levels = np.asarray(getattr(model, "quantile_levels", ()) or (), dtype=float)
    mase_model, mae_model, sql_model, covered, n_points = [], [], [], 0, 0
    for batch, rows in zip(entry["batches"], entry["rows"]):
        inputs, _ = split_batch(to_torch(batch, device))
        out = model(**inputs)
        pred = out["prediction"].float().cpu().numpy()
        quant = out["quantiles"].float().cpu().numpy() if "quantiles" in out else None
        for r in rows:
            e = mae(r["y"], pred[r["b"], r["c"], : r["H"]], r["mask"])
            mae_model.append(e)
            mase_model.append(e / r["scale"])
            if quant is not None:
                q = quant[r["b"], r["c"], : r["H"]][r["mask"]]  # [n, Q]
                y = r["y"][r["mask"]][:, None]
                pinball = np.maximum(levels * (y - q), (levels - 1) * (y - q)).mean()
                sql_model.append(2 * pinball / r["scale"])
                covered += int(((y[:, 0] >= q[:, 0]) & (y[:, 0] <= q[:, -1])).sum())
                n_points += len(y)
    model.train(was_training)

    model_scores = np.array(mase_model)
    base = entry["baselines"]
    summary = {
        "suite": Path(suite_dir).name,
        "n_series": int(len(model_scores)),
        "mae_model": float(np.mean(mae_model)),
        "mase_model": float(np.median(model_scores)),
        **{f"mase_{k}": float(np.median(v)) for k, v in base.items()},
        "win_rate_vs_best": float(np.mean(model_scores < base["best"])),
    }
    if sql_model:
        summary["sql_model"] = float(np.median(sql_model))
        summary["coverage"] = covered / max(n_points, 1)  # of [lowest, highest] quantile
        summary["coverage_nominal"] = float(levels[-1] - levels[0])
    by_family: dict[str, dict[str, float]] = defaultdict(dict)
    for fam in np.unique(entry["families"]):
        sel = entry["families"] == fam
        by_family[str(fam)] = {
            "n": int(sel.sum()),
            "mase_model": float(np.median(model_scores[sel])),
            **{f"mase_{k}": float(np.median(v[sel])) for k, v in base.items()},
        }
    summary["by_family"] = dict(by_family)
    return summary
