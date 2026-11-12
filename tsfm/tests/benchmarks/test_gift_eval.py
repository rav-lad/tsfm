"""GIFT-Eval re-implementation: frequency rules, windows, metrics, model wrapper.

The end-to-end check against the official Seasonal_Naive scores needs the
downloaded benchmark and runs only if ``data/gift_eval`` exists.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pyarrow as pa
import pytest

from tsfm.benchmarks.gift_eval import (
    QUANTILES,
    GiftConfig,
    all_configs,
    get_seasonality,
    load_reference,
    load_task,
    parse_freq,
    run,
    score,
    seasonal_naive,
)
from tsfm.benchmarks.predictor import model_predictor
from tsfm.models import V1_QUANTILES, PatchTransformerV0, PatchV0Config

GIFT_ROOT = Path(__file__).parents[2] / "data" / "gift_eval"


@pytest.mark.parametrize("freq,expected", [
    ("15T", (15, "T")), ("W-SUN", (1, "W")), ("A-DEC", (1, "A")), ("ME", (1, "M")),
    ("5min", (5, "T")), ("h", (1, "H")), ("10S", (10, "S")), ("QE-DEC", (1, "Q")),
])
def test_parse_freq(freq, expected):
    assert parse_freq(freq) == expected


@pytest.mark.parametrize("freq,season", [("H", 24), ("15T", 96), ("5T", 288), ("10T", 144), ("D", 1),
                                         ("W-SUN", 1), ("M", 12), ("Q-DEC", 4), ("A-DEC", 1), ("7T", 1)])
def test_seasonality_matches_gluonts(freq, season):
    assert get_seasonality(freq) == season


def test_97_configs_match_official_names():
    names = {c.config_name for c in all_configs()}
    assert len(names) == 97
    assert names == set(load_reference("Seasonal_Naive"))


def _write_dataset(root: Path, name: str, freq: str, targets: list) -> None:
    d = root / name
    d.mkdir(parents=True)
    table = pa.table({"item_id": [str(i) for i in range(len(targets))], "freq": [freq] * len(targets),
                      "target": targets})
    with pa.ipc.new_stream(d / "data-00000-of-00001.arrow", table.schema) as w:
        w.write_table(table)


def test_windows_and_multivariate_split(tmp_path):
    # daily -> pl 30; min length 1000 -> ceil(0.1 * 1000 / 30) = 4 windows
    T = 1000
    _write_dataset(tmp_path, "toy", "D", [[[float(t) for t in range(T)], [2.0 * t for t in range(T)]]])
    task = load_task(tmp_path, GiftConfig("toy", "short", "toy", "D", "test", 2))
    assert (task.prediction_length, task.windows, task.seasonality) == (30, 4, 1)
    assert len(task.contexts) == 2 * 4  # two variates x four windows
    assert [len(c) for c in task.contexts[:4]] == [T - 120, T - 90, T - 60, T - 30]
    np.testing.assert_array_equal(task.labels[3], np.arange(T - 30, T, dtype=np.float32))
    np.testing.assert_array_equal(task.labels[4], 2 * np.arange(T - 120, T - 90, dtype=np.float32))


def test_mase_masks_constant_contexts_and_missing_labels(tmp_path):
    _write_dataset(tmp_path, "toy", "D", [[float(t) for t in range(400)], [5.0] * 400])
    task = load_task(tmp_path, GiftConfig("toy", "short", "toy", "D", "test", 1))
    fc = task.labels + 2.0  # error 2 everywhere; series 0 has seasonal error 1
    task.labels[0, :5] = np.nan
    res = score(task, fc)
    assert res["MASE"] == pytest.approx(2.0)  # constant series (seasonal error 0) is masked out
    assert math.isfinite(res["ND"])


def test_model_predictor_shapes_and_scale():
    pred = model_predictor(PatchTransformerV0(PatchV0Config(n_layers=1)), batch_size=3)
    rng = np.random.default_rng(0)
    contexts = [rng.normal(1e4, 10, n).astype(np.float32) for n in (20, 300, 2000, 50)]
    contexts[1][10:40] = np.nan
    out = pred(contexts, 250, 1)  # > max_horizon: rollout
    assert out.shape == (4, 250) and np.isfinite(out).all()
    assert np.abs(out - 1e4).max() < 1e3  # denormalized back to the series scale


@pytest.mark.skipif(not GIFT_ROOT.exists(), reason="GIFT-Eval not downloaded")
def test_seasonal_naive_reproduces_official_scores():
    ref = load_reference("Seasonal_Naive")
    configs = [c for c in all_configs() if c.name in ("m4_hourly", "ett1/H", "bitbrains_rnd/H", "car_parts_with_missing")]
    for r in run(GIFT_ROOT, seasonal_naive, configs, log=lambda s: None):
        assert r["MASE"] == pytest.approx(ref[r["dataset"]]["MASE"], rel=1e-4), r["dataset"]
        # ND differs by up to ~0.06% on bitbrains (huge values), MASE matches to 1e-8: likely float32 sums in gluonts
        assert r["ND"] == pytest.approx(ref[r["dataset"]]["ND"], rel=1e-3), r["dataset"]


def test_wql_with_quantiles(tmp_path):
    _write_dataset(tmp_path, "toy", "D", [[float(t) for t in range(400)]])
    task = load_task(tmp_path, GiftConfig("toy", "short", "toy", "D", "test", 1))
    point = task.labels + 1.0
    degenerate = score(task, point, np.repeat(point[..., None], len(QUANTILES), -1))
    assert degenerate["WQL"] == pytest.approx(degenerate["ND"])
    assert score(task, point)["WQL"] == pytest.approx(degenerate["WQL"])
    spread = point[..., None] + (QUANTILES - 0.5) * 4.0  # wide interval around a biased median
    perfect = score(task, task.labels, np.repeat(task.labels[..., None], len(QUANTILES), -1))
    assert perfect["WQL"] == pytest.approx(0.0)
    assert score(task, point, spread)["MASE"] == degenerate["MASE"]  # MASE uses the median only


def test_quantile_model_predictor_returns_median_and_quantiles():
    pred = model_predictor(PatchTransformerV0(PatchV0Config(n_layers=1, quantiles=V1_QUANTILES)), batch_size=2)
    contexts = [np.random.default_rng(i).normal(100, 5, 300).astype(np.float32) for i in range(3)]
    median, q = pred(contexts, 50, 1)
    assert median.shape == (3, 50) and q.shape == (3, 50, 9)
    np.testing.assert_allclose(median, q[..., 4], rtol=1e-6)
    assert (np.diff(q, axis=-1) >= -1e-4).all()


def test_predictor_rejects_other_quantile_levels():
    with pytest.raises(ValueError, match="GIFT-Eval levels"):
        model_predictor(PatchTransformerV0(PatchV0Config(n_layers=1, quantiles=(0.25, 0.5, 0.75))))
