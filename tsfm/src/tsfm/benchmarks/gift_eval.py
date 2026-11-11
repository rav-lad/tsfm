"""GIFT-Eval benchmark (Aksu et al., 2024), re-implemented without gluonts.

The official harness (github.com/SalesforceAIResearch/gift-eval, Apache-2.0)
pins numpy 1.26 / gluonts 0.15. This module reproduces its protocol on our
stack, following ``src/gift_eval/data.py`` and ``notebooks/naive.ipynb`` at
commit 69c919e:

- 97 configurations = 55 short-term datasets + 21 datasets also in medium/long.
- ``prediction_length`` = base length of the frequency (``PRED_LENGTH_MAP``, M4
  has its own map) x 1 / 10 / 15 for short / medium / long.
- test windows: the last ``windows * prediction_length`` points, cut in
  non-overlapping rolling windows; ``windows = min(max(1, ceil(0.1 * min_len /
  pl)), 20)``, M4 uses one window.
- multivariate datasets are split into univariate series.
- MASE[0.5] (gluonts ``MASE``, ``axis=None``): mean over every valid label point
  of ``|y - yhat| / seasonal_error``; ``seasonal_error`` is the nan-mean of
  ``|y_t - y_{t-m}|`` over the whole context with ``m = get_seasonality(freq)``.
  gluonts works on masked arrays, so windows whose seasonal error is 0 or
  undefined (constant / all-missing context) are *masked out*, not infinite.
- ND[0.5] = sum |e| / sum |y|; WQL (the "CRPS" column) = mean over the
  quantiles 0.1..0.9 of 2 * sum pinball / sum |y|. A predictor returning
  ``(median, quantiles)`` gets the real WQL; for a point forecast every quantile
  equals the point, so WQL == ND (a degenerate, pessimistic CRPS).

Validated by reproducing the official Seasonal_Naive scores
(``benchmarks/gift_eval/reference/``).
"""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterator

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc

SHORT_DATASETS = (
    "m4_yearly m4_quarterly m4_monthly m4_weekly m4_daily m4_hourly electricity/15T electricity/H "
    "electricity/D electricity/W solar/10T solar/H solar/D solar/W hospital covid_deaths us_births/D "
    "us_births/M us_births/W saugeenday/D saugeenday/M saugeenday/W temperature_rain_with_missing "
    "kdd_cup_2018_with_missing/H kdd_cup_2018_with_missing/D car_parts_with_missing restaurant "
    "hierarchical_sales/D hierarchical_sales/W LOOP_SEATTLE/5T LOOP_SEATTLE/H LOOP_SEATTLE/D SZ_TAXI/15T "
    "SZ_TAXI/H M_DENSE/H M_DENSE/D ett1/15T ett1/H ett1/D ett1/W ett2/15T ett2/H ett2/D ett2/W "
    "jena_weather/10T jena_weather/H jena_weather/D bitbrains_fast_storage/5T bitbrains_fast_storage/H "
    "bitbrains_rnd/5T bitbrains_rnd/H bizitobs_application bizitobs_service bizitobs_l2c/5T bizitobs_l2c/H"
).split()
MED_LONG_DATASETS = (
    "electricity/15T electricity/H solar/10T solar/H kdd_cup_2018_with_missing/H LOOP_SEATTLE/5T "
    "LOOP_SEATTLE/H SZ_TAXI/15T M_DENSE/H ett1/15T ett1/H ett2/15T ett2/H jena_weather/10T jena_weather/H "
    "bitbrains_fast_storage/5T bitbrains_rnd/5T bizitobs_application bizitobs_service bizitobs_l2c/5T "
    "bizitobs_l2c/H"
).split()
TERMS = {"short": 1, "medium": 10, "long": 15}

PRED_LENGTH_MAP = {"M": 12, "W": 8, "D": 30, "H": 48, "T": 48, "S": 60}
M4_PRED_LENGTH_MAP = {"A": 6, "Q": 8, "M": 18, "W": 13, "D": 14, "H": 48}
TEST_SPLIT, MAX_WINDOW = 0.1, 20
# gluonts.time_feature.get_seasonality defaults
DEFAULT_SEASONALITIES = {"S": 3600, "T": 1440, "H": 24, "D": 1, "W": 1, "M": 12, "B": 5, "Q": 4}
NEW_TO_OLD_FREQ = {"Y": "A", "h": "H", "min": "T", "s": "S", "us": "U"}
PRETTY_NAMES = {
    "saugeenday": "saugeen",
    "temperature_rain_with_missing": "temperature_rain",
    "kdd_cup_2018_with_missing": "kdd_cup_2018",
    "car_parts_with_missing": "car_parts",
}
QUANTILES = np.arange(1, 10) / 10

REFERENCE_DIR = Path(__file__).parents[3] / "benchmarks" / "gift_eval" / "reference"

# contexts (list of 1D float arrays, NaN = missing), prediction_length, seasonality
#   -> point forecast [n, pl], or (median [n, pl], quantiles [n, pl, 9] at QUANTILES)
Predictor = Callable[[list[np.ndarray], int, int], "np.ndarray | tuple[np.ndarray, np.ndarray]"]


# ---------------------------------------------------------------------- frequency
def parse_freq(freq: str) -> tuple[int, str]:
    """``"15T" -> (15, "T")``, ``"W-SUN" -> (1, "W")``, ``"ME" -> (1, "M")``, ``"5min" -> (5, "T")``."""
    m = re.fullmatch(r"(\d*)([A-Za-z]+)(-.*)?", freq)
    if m is None:
        raise ValueError(f"cannot parse frequency {freq!r}")
    n = int(m.group(1) or 1)
    base = m.group(2)
    base = NEW_TO_OLD_FREQ.get(base, base)
    if len(base) >= 2 and base[-1] in "SE":  # MS / ME / QE / YE ... (gluonts norm_freq_str)
        base = base[:-1]
    return n, NEW_TO_OLD_FREQ.get(base, base)


def get_seasonality(freq: str) -> int:
    n, base = parse_freq(freq)
    season, rem = divmod(DEFAULT_SEASONALITIES.get(base, 1), n)
    return season if not rem else 1


# ---------------------------------------------------------------------- data
def _rows(arr: pa.Array) -> list[np.ndarray]:
    """Arrow list (of lists) column -> one float32 array per row, nulls as NaN."""
    lens = pc.list_value_length(arr).to_numpy(zero_copy_only=False).astype(np.int64)
    child = arr.flatten()
    if pa.types.is_list(child.type) or pa.types.is_large_list(child.type) or pa.types.is_fixed_size_list(child.type):
        inner = _rows(child)
        bounds = np.r_[0, np.cumsum(lens)]
        return [np.stack(inner[a:b]) for a, b in zip(bounds[:-1], bounds[1:])]
    values = child.to_numpy(zero_copy_only=False).astype(np.float32)
    return np.split(values, np.cumsum(lens)[:-1])


@dataclass
class GiftConfig:
    name: str  # directory name, e.g. "ett1/H"
    term: str  # short / medium / long
    key: str  # pretty dataset key, e.g. "ett1"
    freq_label: str  # frequency as written in the official config names
    domain: str
    num_variates: int

    @property
    def config_name(self) -> str:  # same string as the official all_results.csv
        return f"{self.key}/{self.freq_label}/{self.term}"


def all_configs(properties: dict | None = None) -> list[GiftConfig]:
    props = properties or json.loads((REFERENCE_DIR / "dataset_properties.json").read_text())
    out = []
    for ds in dict.fromkeys(SHORT_DATASETS + MED_LONG_DATASETS):
        if "/" in ds:
            key, freq_label = ds.split("/")
            key = PRETTY_NAMES.get(key.lower(), key.lower())
        else:
            key = PRETTY_NAMES.get(ds.lower(), ds.lower())
            freq_label = props[key]["frequency"]
        terms = (["short"] if ds in SHORT_DATASETS else []) + (["medium", "long"] if ds in MED_LONG_DATASETS else [])
        for term in terms:
            out.append(GiftConfig(ds, term, key, freq_label, props[key]["domain"], props[key]["num_variates"]))
    return out


@dataclass
class GiftTask:
    config: GiftConfig
    freq: str
    prediction_length: int
    seasonality: int
    windows: int
    contexts: list[np.ndarray]
    labels: np.ndarray  # [n, pl]
    item_ids: list[str]


def load_task(root: str | Path, cfg: GiftConfig) -> GiftTask:
    path = Path(root) / cfg.name
    files = sorted(path.glob("data-*.arrow"))
    if not files:
        raise FileNotFoundError(f"{path}: no arrow file (download Salesforce/GiftEval first)")
    table = pa.concat_tables(pa.ipc.open_stream(f).read_all() for f in files)
    freq = table.column("freq")[0].as_py()
    targets = _rows(table.column("target").combine_chunks())
    ids = table.column("item_id").to_pylist()

    _, base = parse_freq(freq)
    base_len = M4_PRED_LENGTH_MAP[base] if "m4" in cfg.name else PRED_LENGTH_MAP[base]
    pl = TERMS[cfg.term] * base_len
    min_len = min(t.shape[-1] for t in targets)  # first variate == all variates
    windows = 1 if "m4" in cfg.name else min(max(1, math.ceil(TEST_SPLIT * min_len / pl)), MAX_WINDOW)

    series, series_ids = [], []
    for item, t in zip(ids, targets):
        if t.ndim == 1:
            series.append(t)
            series_ids.append(item)
        else:  # MultivariateToUnivariate
            for d, row in enumerate(t):
                series.append(row)
                series_ids.append(f"{item}_dim{d}")

    contexts, labels, inst_ids = [], [], []
    for item, y in zip(series_ids, series):
        T = len(y)
        for w in range(windows):
            cut = T - pl * (windows - w)
            contexts.append(y[:cut])
            labels.append(y[cut : cut + pl])
            inst_ids.append(f"{item}@{cut}")
    return GiftTask(cfg, freq, pl, get_seasonality(freq), windows, contexts, np.stack(labels), inst_ids)


# ---------------------------------------------------------------------- metrics
def seasonal_error(context: np.ndarray, m: int) -> float:
    lag = m if m < len(context) else 1
    d = np.abs(context[:-lag] - context[lag:])
    d = d[np.isfinite(d)]
    return float(d.mean()) if d.size else math.nan


def score(task: GiftTask, forecasts: np.ndarray, quantiles: np.ndarray | None = None) -> dict[str, float]:
    """MASE / MAE / ND of the point (median) forecast; WQL of ``quantiles`` [n, pl, 9]
    (levels ``QUANTILES``) or, without them, of the point used as every quantile."""
    y = task.labels.astype(np.float64)
    f = forecasts.astype(np.float64)
    if f.shape != y.shape:
        raise ValueError(f"forecast shape {f.shape} != labels {y.shape}")
    if not np.isfinite(f).all():
        raise ValueError(f"{task.config.config_name}: non-finite forecasts")
    valid = np.isfinite(y)
    se = np.array([seasonal_error(c.astype(np.float64), task.seasonality) for c in task.contexts])
    err = np.abs(y - f)
    scalable = np.isfinite(se) & (se > 0)  # np.ma masks x / 0 and masked divisors
    with np.errstate(divide="ignore", invalid="ignore"):
        ase = err / se[:, None]
    ase_valid = valid & scalable[:, None]
    abs_y = np.abs(np.where(valid, y, 0.0)).sum()
    e = np.where(valid, err, 0.0)
    if quantiles is None:
        q = f[..., None]
    else:
        q = quantiles.astype(np.float64)
        if q.shape != y.shape + (len(QUANTILES),) or not np.isfinite(q).all():
            raise ValueError(f"{task.config.config_name}: bad quantile forecast {q.shape}")
    diff = np.where(valid[..., None], y[..., None] - q, 0.0)
    pinball = np.maximum(QUANTILES * diff, (QUANTILES - 1) * diff).sum(axis=(0, 1))
    return {
        "MASE": float(ase[ase_valid].mean()),
        "MAE": float(e.sum() / valid.sum()),
        "ND": float(e.sum() / abs_y),
        "WQL": float(np.mean(2 * pinball / abs_y)),
    }


# ---------------------------------------------------------------------- baselines
def last_value_imputation(x: np.ndarray) -> np.ndarray:
    """gluonts LastValueImputation: forward fill, leading NaNs -> 0."""
    x = x.astype(np.float64)
    nan = np.isnan(x)
    if not nan.any():
        return x
    if nan.all():
        return np.zeros_like(x)
    idx = np.where(~nan, np.arange(len(x)), 0)
    np.maximum.accumulate(idx, out=idx)
    out = x[idx]
    out[: int(np.argmax(~nan))] = 0.0
    return out


def naive(contexts: list[np.ndarray], pl: int, season: int) -> np.ndarray:
    return np.stack([np.full(pl, last_value_imputation(c)[-1]) for c in contexts])


def seasonal_naive(contexts: list[np.ndarray], pl: int, season: int) -> np.ndarray:
    out = []
    for c in contexts:
        x = last_value_imputation(c)
        m = season if season <= len(x) else 1
        out.append(x[-m:][np.arange(pl) % m])
    return np.stack(out)


BASELINES: dict[str, Predictor] = {"naive": naive, "seasonal_naive": seasonal_naive}


# ---------------------------------------------------------------------- run + report
def run(root: str | Path, predictor: Predictor, configs: list[GiftConfig] | None = None,
        log: Callable[[str], None] = print) -> Iterator[dict]:
    for cfg in configs or all_configs():
        task = load_task(root, cfg)
        fc = predictor(task.contexts, task.prediction_length, task.seasonality)
        fc, fq = fc if isinstance(fc, tuple) else (fc, None)
        res = {"dataset": cfg.config_name, **score(task, fc, fq), "n_instances": len(task.contexts),
               "prediction_length": task.prediction_length, "domain": cfg.domain,
               "num_variates": cfg.num_variates}
        log(f"{cfg.config_name:<40} n={len(task.contexts):<7} pl={task.prediction_length:<5} "
            f"MASE {res['MASE']:.3f}  ND {res['ND']:.3f}")
        yield res


def load_reference(model: str) -> dict[str, dict[str, float]]:
    """Official per-config scores from the GIFT-Eval repo (``reference/<model>.csv``)."""
    import csv

    with open(REFERENCE_DIR / f"{model}.csv", encoding="utf-8") as f:
        return {
            r["dataset"]: {"MASE": float(r["eval_metrics/MASE[0.5]"]),
                           "WQL": float(r["eval_metrics/mean_weighted_sum_quantile_loss"]),
                           "ND": float(r["eval_metrics/ND[0.5]"])}
            for r in csv.DictReader(f)
        }


def relative_gmean(scores: dict[str, float], baseline: dict[str, float]) -> float:
    """Leaderboard-style aggregate: geometric mean over configs of score / seasonal-naive score."""
    keys = [k for k in scores if k in baseline]
    return float(np.exp(np.mean([np.log(scores[k] / baseline[k]) for k in keys])))
