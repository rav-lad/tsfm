"""Build and freeze the synthetic evaluation suites (sprint D4).

    uv run python scripts/build_holdouts.py                      # all val_* suites
    uv run python scripts/build_holdouts.py --suites test_syn_id --include-test
    uv run python scripts/build_holdouts.py --n 200 --out /tmp/x # quick dry run

Suites already present are not overwritten unless --force: a frozen suite must
stay frozen. test_syn_* suites are skipped unless --include-test.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

from tsfm.data.holdouts import build_suite, suite_configs


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--holdout", default="configs/holdout_v1.yaml")
    ap.add_argument("--out", default="data/holdouts/holdout_v1")
    ap.add_argument("--suites", nargs="*")
    ap.add_argument("--include-test", action="store_true")
    ap.add_argument("--n", type=int, help="override suite sizes (dry runs only)")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    spec, suites = suite_configs(args.holdout)
    names = args.suites or [s for s in suites if args.include_test or not s.startswith("test_syn")]
    for name in names:
        target = Path(args.out) / name / "manifest.json"
        if target.exists() and not args.force:
            print(f"skip {name}: already frozen ({target})")
            continue
        t0 = time.perf_counter()
        m = build_suite(name, suites[name], spec, args.out, n=args.n)
        dt = time.perf_counter() - t0
        print(f"{name:<20} n={m['n_episodes']:<6} {dt:6.1f}s  content={m['content_sha256'][:12]}  -> {Path(args.out) / name}")


if __name__ == "__main__":
    main()
