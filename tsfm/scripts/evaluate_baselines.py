"""Reference scores of simple baselines on frozen suites (reference §12.3, §13.1).

These numbers are the bar any trained model must clear, per suite and family.
For multivariate suites it also reports the relational diagnostics
(covariate lift for KNOWN_FUTURE, history dependence for PAST_ONLY) by
relation type: useful relations must show signal, decoys must not.

    uv run python scripts/evaluate_baselines.py --root data/holdouts/holdout_v1
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

import numpy as np

from tsfm.data.evaluation import evaluate_baselines, relation_table
from tsfm.data.frozen import FrozenEpisodeSet

BASELINES = ("naive", "seasonal_naive", "drift", "mean", "best")


def _fmt(v: list[float]) -> str:
    return f"{np.median(v):.2f}" if v else "-"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="data/holdouts/holdout_v1")
    ap.add_argument("--suites", nargs="*")
    ap.add_argument("--out", default="reports/data/baselines_holdout_v1.md")
    args = ap.parse_args()

    root = Path(args.root)
    names = args.suites or sorted(p.name for p in root.iterdir() if (p / "manifest.json").exists() and not p.name.startswith("test_syn"))
    lines = [f"# Baseline reference scores — {root}", "", "Median MASE on the leak-free normalized scale (lower is better).", ""]
    for name in names:
        ds = FrozenEpisodeSet(root / name)
        by_fam: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
        rel: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
        for ep in ds:
            fam = ep.meta["family"]
            for row in evaluate_baselines(ep):
                for b in BASELINES:
                    by_fam[fam][b].append(row[f"mase_{b}"])
                    by_fam["ALL"][b].append(row[f"mase_{b}"])
            for r in relation_table(ep):
                key = f"{r['role']}:{'edge-' + r['strength'] if r['edge'] else r['op']}"
                rel[key]["history_dependence"].append(r["history_dependence"])
                if "lift" in r:
                    rel[key]["lift"].append(r["lift"])
        m = ds.manifest
        lines += [
            f"## {name}  (split `{m['split']}`, n={m['n_episodes']}, content `{m['content_sha256'][:12]}`)",
            "",
            "| family | n | " + " | ".join(BASELINES) + " |",
            "|---|---:|" + "---:|" * len(BASELINES),
        ]
        for fam in sorted(by_fam, key=lambda f: (f == "ALL", f)):
            d = by_fam[fam]
            lines.append(f"| {fam} | {len(d['best'])} | " + " | ".join(_fmt(d[b]) for b in BASELINES) + " |")
        if rel:
            lines += ["", "| relation (role:type) | n pairs | median lift (KF) | median history dependence |", "|---|---:|---:|---:|"]
            for key in sorted(rel):
                d = rel[key]
                lines.append(f"| {key} | {len(d['history_dependence'])} | {_fmt(d['lift'])} | {_fmt(d['history_dependence'])} |")
        lines.append("")
        print(f"done {name}")
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
