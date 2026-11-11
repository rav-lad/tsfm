"""Zero-shot evaluation on GIFT-Eval (97 configs), comparable to the official leaderboard.

    # once: download the benchmark (1.6 GB) to data/gift_eval
    uv run python scripts/eval_gift.py --download
    uv run python scripts/eval_gift.py --model seasonal_naive         # protocol check vs official scores
    uv run python scripts/eval_gift.py --model runs/v0/best.pt        # our model
    uv run python scripts/eval_gift.py --model runs/v0/best.pt --datasets m4 ett1 --terms short   # quick subset

Writes reports/gift_eval/<name>.csv (one row per config) and <name>.md
(leaderboard-style aggregate: geometric mean of MASE / Seasonal-Naive MASE,
next to the official scores of reference models on the same configs).

The model is point-forecast only: its WQL ("CRPS" column) treats the point as
every quantile, which equals ND and is pessimistic. Compare models on MASE.
"""

from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path

from tsfm.benchmarks.gift_eval import BASELINES, all_configs, load_reference, relative_gmean, run

REFERENCES = ("Seasonal_Naive", "Naive", "chronos_bolt_small", "Timestral-5M", "Timestral-28M")
FIELDS = ("dataset", "MASE", "ND", "WQL", "MAE", "n_instances", "prediction_length", "domain", "num_variates")


def _predictor(args):
    if args.model in BASELINES:
        return BASELINES[args.model], args.model
    from tsfm.benchmarks.predictor import model_predictor
    from tsfm.training.trainer import load_model, resolve_device

    device = resolve_device(args.device)
    model = load_model(args.model, device)
    name = args.name or "_".join(Path(args.model).with_suffix("").parts[-2:])
    print(f"{args.model}: {model.num_parameters():,} params on {device}")
    return model_predictor(model, device, args.batch_size, args.max_context), name


def _report(name: str, rows: list[dict], partial: bool) -> str:
    ours = {r["dataset"]: r["MASE"] for r in rows}
    refs = {m: load_reference(m) for m in REFERENCES}
    snaive = {k: v["MASE"] for k, v in refs["Seasonal_Naive"].items()}
    lines = [f"# GIFT-Eval — {name}", ""]
    if partial:
        lines += [f"**Subset: {len(rows)} / 97 configs.** Not comparable to the leaderboard aggregate.", ""]
    lines += ["Geometric mean over configs of MASE / Seasonal-Naive MASE (lower is better; < 1 beats seasonal naive).", "",
              "| model | all | short | medium | long |", "|---|---:|---:|---:|---:|"]

    def agg(scores: dict[str, float]) -> list[str]:
        cells = []
        for term in (None, "short", "medium", "long"):
            sub = {k: v for k, v in scores.items() if k in ours and (term is None or k.endswith("/" + term))}
            cells.append(f"{relative_gmean(sub, snaive):.3f}" if sub else "-")
        return cells

    lines.append(f"| **{name}** | " + " | ".join(agg(ours)) + " |")
    for m, ref in refs.items():
        lines.append(f"| {m} (official) | " + " | ".join(agg({k: v['MASE'] for k, v in ref.items()})) + " |")
    lines += ["", "| config | pl | MASE ours | Seasonal Naive | Timestral-5M | chronos_bolt_small |", "|---|---:|---:|---:|---:|---:|"]
    for r in rows:
        k = r["dataset"]
        lines.append(f"| {k} | {r['prediction_length']} | {r['MASE']:.3f} | {snaive[k]:.3f} | "
                     f"{refs['Timestral-5M'][k]['MASE']:.3f} | {refs['chronos_bolt_small'][k]['MASE']:.3f} |")
    return "\n".join(lines) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="seasonal_naive", help="seasonal_naive | naive | path to a checkpoint")
    ap.add_argument("--root", default="data/gift_eval")
    ap.add_argument("--download", action="store_true", help="download Salesforce/GiftEval to --root and exit")
    ap.add_argument("--datasets", nargs="*", help="keep configs whose name contains one of these strings")
    ap.add_argument("--terms", nargs="*", choices=["short", "medium", "long"])
    ap.add_argument("--name")
    ap.add_argument("--out-dir", default="reports/gift_eval")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--max-context", type=int, default=1024)
    args = ap.parse_args()

    if args.download:
        from huggingface_hub import snapshot_download

        print(snapshot_download("Salesforce/GiftEval", repo_type="dataset", local_dir=args.root))
        return

    configs = [c for c in all_configs()
               if (not args.datasets or any(d in c.config_name for d in args.datasets))
               and (not args.terms or c.term in args.terms)]
    predictor, name = _predictor(args)
    t0 = time.perf_counter()
    rows = list(run(args.root, predictor, configs))
    print(f"{len(rows)} configs in {time.perf_counter() - t0:.0f}s")

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    with open(out / f"{name}.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    report = _report(name, rows, partial=len(rows) < 97)
    (out / f"{name}.md").write_text(report, encoding="utf-8")
    print("\n".join(report.splitlines()[:12]))
    print(f"-> {out / (name + '.csv')}, {out / (name + '.md')}")


if __name__ == "__main__":
    main()
