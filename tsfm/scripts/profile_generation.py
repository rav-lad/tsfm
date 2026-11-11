"""Measure generation throughput (series/s) per family and context length.

    uv run python scripts/profile_generation.py --n 300
"""

from __future__ import annotations

import argparse
import time

import numpy as np

from tsfm.data import SyntheticEpisodeSource, collate_episodes, load_config


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/data_v0.yaml")
    ap.add_argument("--n", type=int, default=300)
    args = ap.parse_args()
    cfg = load_config(args.config)

    print(f"{'family':<18}{'L':>6}{'H':>6}{'ep/s':>10}{'p50 ms':>9}{'p99 ms':>9}")
    for L in (128, 512, 1024):
        cfg_l = {**cfg, "sequence": {**cfg["sequence"], "context_lengths": {L: 1.0}, "horizons": {96: 1.0}}}
        src = SyntheticEpisodeSource(cfg_l, split="debug")
        for fam in src.mix:
            times = []
            for i in range(args.n):
                t0 = time.perf_counter()
                src.episode(i, family=fam)
                times.append(time.perf_counter() - t0)
            ms = np.array(times) * 1e3
            print(f"{fam:<18}{L:>6}{96:>6}{1e3 / ms.mean():>10.0f}{np.median(ms):>9.2f}{np.percentile(ms, 99):>9.2f}")

    src = SyntheticEpisodeSource(cfg, split="debug")
    t0 = time.perf_counter()
    eps = [src.episode(i) for i in range(2000)]
    t_gen = time.perf_counter() - t0
    t0 = time.perf_counter()
    for k in range(0, 2000, 64):
        collate_episodes(eps[k : k + 64], pad_to_multiple=32)
    t_col = time.perf_counter() - t0
    print(f"\nfull mix: {2000 / t_gen:.0f} episodes/s/core generation, {2000 / t_col:.0f} episodes/s collate")
    print(f"=> ~{2000 / (t_gen + t_col):.0f} episodes/s per DataLoader worker (single core)")


if __name__ == "__main__":
    main()
