"""Plot grids of episodes per family (history in colour, hidden future in grey).

    uv run python scripts/inspect_samples.py --n 24 --out reports/data/samples
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from tsfm.data import SyntheticEpisodeSource, load_config


def plot_family(src: SyntheticEpisodeSource, family: str, n: int, out: Path, start: int) -> None:
    cols = 4
    rows = -(-n // cols)
    fig, axes = plt.subplots(rows, cols, figsize=(4.2 * cols, 2.2 * rows), squeeze=False)
    for k, ax in enumerate(axes.flat):
        if k >= n:
            ax.axis("off")
            continue
        ep = src.episode(start + k, family=family)
        L, H = ep.context_length, ep.prediction_length
        x = ep.values[0].astype(float)
        obs = ep.observed_mask[0]
        t = np.arange(L + H)
        hist = np.where(obs[:L], x[:L], np.nan)
        ax.plot(t[:L], hist, lw=0.8, color="#2b6cb0")
        ax.plot(t[L:], x[L:], lw=0.8, color="#a0a0a0")
        miss = ~obs[:L]
        if miss.any():
            ax.scatter(t[:L][miss], np.full(miss.sum(), np.nanmin(hist)), s=2, color="#e53e3e")
        ax.axvline(L, color="k", lw=0.5, ls=":")
        aug = ",".join(a["op"] for a in ep.meta.get("augmentations", [])) or "-"
        ax.set_title(f"{ep.meta['kind']} | L={L} H={H} | aug={aug}", fontsize=7)
        ax.tick_params(labelsize=6)
    fig.suptitle(f"{family}  (blue=visible history, grey=hidden future, red=missing)", fontsize=10)
    fig.tight_layout()
    fig.savefig(out / f"{family}.png", dpi=110)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/data_v0.yaml")
    ap.add_argument("--split", default="debug")
    ap.add_argument("--n", type=int, default=24)
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--out", default="reports/data/samples")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    src = SyntheticEpisodeSource(load_config(args.config), split=args.split)
    for fam in src.mix:
        plot_family(src, fam, args.n, out, args.start)
        print("wrote", out / f"{fam}.png")


if __name__ == "__main__":
    main()
