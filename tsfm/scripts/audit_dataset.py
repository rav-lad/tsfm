"""Distribution and difficulty audit of the episode mix.

For N episodes, reports per family/kind: counts, augmentation and missingness
rates, scale fallbacks, lag-1 ACF, zero-rate, and the normalized MAE of simple
baselines (last value, seasonal naive with the best lag in the visible history,
linear drift). Baselines are computed on the normalized scale, so numbers are
comparable across families.

    uv run python scripts/audit_dataset.py --n 5000 --out reports/data/audit_v0.md
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

import numpy as np

from tsfm.data import RobustScaler, SyntheticEpisodeSource, load_config


def _baselines(hist: np.ndarray, H: int) -> dict[str, np.ndarray]:
    L = len(hist)
    last = np.full(H, hist[-1])
    # seasonal naive: lag with the highest ACF among 2..L/2 on the visible history
    x = hist - hist.mean()
    denom = (x**2).sum() or 1.0
    max_lag = max(2, min(L // 2, 400))
    acf = np.array([(x[:-k] * x[k:]).sum() / denom for k in range(2, max_lag)])
    lag = int(np.argmax(acf)) + 2 if acf.size else 1
    season = hist[-lag:][np.arange(H) % lag]
    # drift: slope between first and last visible points (Hyndman drift)
    slope = (hist[-1] - hist[0]) / max(L - 1, 1)
    drift = hist[-1] + slope * np.arange(1, H + 1)
    return {"naive": last, "seasonal_naive": season, "drift": drift}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/data_v0.yaml")
    ap.add_argument("--split", default="debug")
    ap.add_argument("--n", type=int, default=5000)
    ap.add_argument("--out", default="reports/data/audit_v0.md")
    args = ap.parse_args()

    src = SyntheticEpisodeSource(load_config(args.config), split=args.split)
    scaler = RobustScaler()
    rows: dict[tuple[str, str], dict[str, list]] = defaultdict(lambda: defaultdict(list))
    for i in range(args.n):
        ep = src.episode(i)
        t = scaler.transform(ep)
        L, H = ep.context_length, ep.prediction_length
        z_hist = t["input_values"][0, :L].astype(float)
        vis = ep.observed_mask[0, :L]
        # baselines need a gap-free history: forward-fill missing points
        idx = np.where(vis, np.arange(L), 0)
        np.maximum.accumulate(idx, out=idx)
        z_hist = z_hist[idx]
        y = t["target_future"][0].astype(float)
        r = rows[(ep.meta["family"], ep.meta["kind"])]
        for name, pred in _baselines(z_hist, H).items():
            r[name].append(np.mean(np.abs(pred - y)))
        raw = ep.values[0, :L][vis].astype(float)
        xc = raw - raw.mean()
        r["acf1"].append(float((xc[:-1] * xc[1:]).sum() / ((xc**2).sum() or 1.0)))
        r["zero_rate"].append(float((raw == 0).mean()))
        r["aug"].append(len(ep.meta["augmentations"]) > 0)
        r["missing"].append(ep.meta["missingness"][0]["kind"] != "complete")
        r["fallback"].append(bool(t["scale_fallback"][0]))
        r["hint"].append(ep.meta["predictability_hint"])

    lines = [
        f"# Data audit — {args.config}, split={args.split}, N={args.n}",
        "",
        f"Effective mix: {', '.join(f'{k}={v:.3f}' for k, v in src.mix.items())}  ",
        f"Not yet implemented (weights redistributed): {', '.join(src.dropped_families) or 'none'}",
        "",
        "MAE on the robust-normalized scale (lower = easier). `best` = best of the three baselines.",
        "",
        "| family | kind | n | naive | s-naive | drift | best | acf1 | zero% | aug% | miss% | fallback% | hints |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for (fam, kind), r in sorted(rows.items()):
        n = len(r["naive"])
        best = np.minimum.reduce([np.array(r[k]) for k in ("naive", "seasonal_naive", "drift")])
        hints = {h: r["hint"].count(h) for h in sorted(set(r["hint"]))}
        lines.append(
            f"| {fam} | {kind} | {n} | {np.median(r['naive']):.2f} | {np.median(r['seasonal_naive']):.2f} | "
            f"{np.median(r['drift']):.2f} | {np.median(best):.2f} | {np.mean(r['acf1']):.2f} | "
            f"{100 * np.mean(r['zero_rate']):.0f} | {100 * np.mean(r['aug']):.0f} | {100 * np.mean(r['missing']):.0f} | "
            f"{100 * np.mean(r['fallback']):.1f} | {hints} |"
        )
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
