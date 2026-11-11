"""Measure DataLoader throughput (episodes/s) as a function of num_workers.

    uv run python scripts/profile_loader.py --workers 0 2 4 8 --batches 40
"""

from __future__ import annotations

import argparse
import time

from tsfm.training import make_train_loader


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/data_v1.yaml")
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--batches", type=int, default=40)
    ap.add_argument("--workers", type=int, nargs="+", default=[0, 2, 4, 8])
    args = ap.parse_args()

    print(f"{'workers':>8}{'batches/s':>12}{'episodes/s':>12}")
    for nw in args.workers:
        loader = make_train_loader(args.config, batch_size=args.batch_size, num_workers=nw)
        it = iter(loader)
        next(it)  # exclude worker start-up
        t0 = time.perf_counter()
        for _ in range(args.batches):
            next(it)
        dt = time.perf_counter() - t0
        print(f"{nw:>8}{args.batches / dt:>12.1f}{args.batches * args.batch_size / dt:>12.0f}")
        del it, loader


if __name__ == "__main__":
    main()
