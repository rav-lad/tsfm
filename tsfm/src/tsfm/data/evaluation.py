"""Reference forecasts, metrics and relational diagnostics (reference §5.4, §12.3, §13.1).

Everything is computed on the leak-free normalized scale (past statistics only),
so scores are comparable across families and scales.

- Point baselines: naive (last value), seasonal naive (lag picked by ACF on the
  visible history), drift, historical mean.
- Metrics: MAE, MASE (scaled by in-sample one-step naive MAE), pinball loss /
  weighted quantile loss and interval coverage for probabilistic models.
- Relational diagnostics:
    * ``covariate_lift``   : does a simple regression using the KNOWN_FUTURE
      covariates forecast the target better than the same regression without
      them? (positive on useful relations, ~0 on decoys)
    * ``history_dependence``: max |cross-correlation| between the differenced
      target history and each covariate history over lags 0..max_lag (works for
      PAST_ONLY drivers, whose future is never visible).
"""

from __future__ import annotations

from typing import Any

import numpy as np

from .episode import Episode, Role
from .normalizers import RobustScaler

# ---------------------------------------------------------------------- metrics


def mae(y: np.ndarray, yhat: np.ndarray, mask: np.ndarray | None = None) -> float:
    err = np.abs(y - yhat)
    return float(err[mask].mean() if mask is not None else err.mean())


def mase_scale(history: np.ndarray) -> float:
    d = np.abs(np.diff(history))
    s = float(d.mean()) if d.size else 0.0
    return s if s > 1e-8 else 1.0


def pinball_loss(y: np.ndarray, q_pred: np.ndarray, quantiles: np.ndarray) -> np.ndarray:
    """Mean pinball loss per quantile. ``q_pred``: [..., Q], ``y``: [...]."""
    diff = y[..., None] - q_pred
    q = np.asarray(quantiles)
    return np.maximum(q * diff, (q - 1) * diff).reshape(-1, len(q)).mean(axis=0)


def weighted_quantile_loss(y: np.ndarray, q_pred: np.ndarray, quantiles: np.ndarray) -> float:
    """GIFT-Eval-style WQL: 2 * sum pinball / sum |y|, averaged over quantiles."""
    diff = y[..., None] - q_pred
    q = np.asarray(quantiles)
    pl = np.maximum(q * diff, (q - 1) * diff).sum(axis=tuple(range(y.ndim)))
    return float(np.mean(2 * pl / max(np.abs(y).sum(), 1e-8)))


def interval_coverage(y: np.ndarray, lower: np.ndarray, upper: np.ndarray) -> float:
    return float(np.mean((y >= lower) & (y <= upper)))


# ---------------------------------------------------------------------- baselines


def _ffill(x: np.ndarray, visible: np.ndarray) -> np.ndarray:
    idx = np.where(visible, np.arange(len(x)), 0)
    np.maximum.accumulate(idx, out=idx)
    out = x[idx]
    if not visible[0]:  # leading gap: back-fill with the first visible value
        first = int(np.argmax(visible))
        out[:first] = x[first]
    return out


def best_seasonal_lag(history: np.ndarray, max_lag: int = 400) -> int:
    x = history - history.mean()
    denom = float((x**2).sum()) or 1.0
    hi = max(3, min(len(x) // 2, max_lag))
    acf = np.array([(x[:-k] * x[k:]).sum() / denom for k in range(2, hi)])
    return int(np.argmax(acf)) + 2 if acf.size else 1


def baseline_forecasts(history: np.ndarray, H: int) -> dict[str, np.ndarray]:
    L = len(history)
    lag = best_seasonal_lag(history)
    slope = (history[-1] - history[0]) / max(L - 1, 1)
    return {
        "naive": np.full(H, history[-1]),
        "seasonal_naive": history[-lag:][np.arange(H) % lag],
        "drift": history[-1] + slope * np.arange(1, H + 1),
        "mean": np.full(H, history.mean()),
    }


def evaluate_baselines(ep: Episode, scaler: RobustScaler | None = None) -> list[dict[str, Any]]:
    """One row per target channel with MAE / MASE of every baseline."""
    t = (scaler or RobustScaler()).transform(ep)
    L, H = ep.context_length, ep.prediction_length
    rows = []
    for c in ep.target_channels:
        hist = _ffill(t["input_values"][c, :L].astype(float), ep.observed_mask[c, :L])
        y = t["target_future"][c].astype(float)
        m = ep.loss_mask[c]
        scale = mase_scale(hist)
        row: dict[str, Any] = {"channel": int(c)}
        for name, pred in baseline_forecasts(hist, H).items():
            e = mae(y, pred, m)
            row[f"mae_{name}"] = e
            row[f"mase_{name}"] = e / scale
        row["mase_best"] = min(row[f"mase_{k}"] for k in ("naive", "seasonal_naive", "drift", "mean"))
        rows.append(row)
    return rows


# ---------------------------------------------------------------------- relational diagnostics


def _design(x_cols: list[np.ndarray], n: int, lags: int) -> np.ndarray:
    t = np.arange(n, dtype=float) / n
    cols = [np.ones(n), t]
    for x in x_cols:
        for k in range(lags + 1):
            cols.append(np.r_[np.full(k, x[0]), x[: n - k]] if k else x)
    return np.stack(cols, axis=1)


def _ridge_forecast(y_hist: np.ndarray, X_all: np.ndarray, L: int, alpha: float = 1e-2) -> np.ndarray:
    Xh = X_all[:L]
    A = Xh.T @ Xh + alpha * np.eye(Xh.shape[1])
    beta = np.linalg.solve(A, Xh.T @ y_hist)
    return X_all[L:] @ beta


def covariate_lift(ep: Episode, target: int, covariates: list[int], lags: int = 3, scaler: RobustScaler | None = None) -> float:
    """Relative MAE improvement from adding KNOWN_FUTURE covariates to a ridge
    forecaster (intercept + time trend + covariate lags 0..``lags``).

    Uses only visible inputs: target history, covariate history and the
    covariate future values that are legitimately known. Returns
    ``1 - MAE_with / MAE_without`` (> 0 means the covariates help).
    """
    for c in covariates:
        if ep.roles[c] != Role.KNOWN_FUTURE:
            raise ValueError("covariate_lift only uses KNOWN_FUTURE covariates")
    t = (scaler or RobustScaler()).transform(ep)
    L, H = ep.context_length, ep.prediction_length
    n = L + H
    y_hist = _ffill(t["input_values"][target, :L].astype(float), ep.observed_mask[target, :L])
    y_fut = t["target_future"][target].astype(float)
    xs = [_ffill(t["input_values"][c].astype(float), ep.observed_mask[c]) for c in covariates]
    f0 = _ridge_forecast(y_hist, _design([], n, lags), L)
    f1 = _ridge_forecast(y_hist, _design(xs, n, lags), L)
    e0, e1 = mae(y_fut, f0), mae(y_fut, f1)
    return float(1.0 - e1 / max(e0, 1e-8))


def history_dependence(ep: Episode, target: int, covariate: int, max_lag: int = 32) -> float:
    """max_{0<=k<=max_lag} |corr(diff y_t, diff x_{t-k})| on the visible history."""
    L = ep.context_length
    y = np.diff(_ffill(ep.values[target, :L].astype(float), ep.observed_mask[target, :L]))
    x = np.diff(_ffill(ep.values[covariate, :L].astype(float), ep.observed_mask[covariate, :L]))
    best = 0.0
    for k in range(0, min(max_lag, len(y) // 4) + 1):
        a, b = y[k:], x[: len(x) - k]
        if a.std() < 1e-12 or b.std() < 1e-12:
            continue
        best = max(best, abs(float(np.corrcoef(a, b)[0, 1])))
    return best


def relation_table(ep: Episode) -> list[dict[str, Any]]:
    """For each (target, covariate) pair: is there an edge, the op, and diagnostics."""
    g = ep.meta.get("graph")
    if g is None:
        return []
    nodes = {n["index"]: n for n in g["nodes"]}
    edges = {(e["parent"], e["child"]): e for e in g["edges"]}
    rows = []
    for tgt in ep.target_channels:
        for cov in range(ep.n_channels):
            if ep.roles[cov] == Role.TARGET:
                continue
            e = edges.get((cov, int(tgt)))
            row = {
                "target": int(tgt),
                "covariate": cov,
                "role": Role(int(ep.roles[cov])).name,
                "edge": e is not None,
                "op": e["op"] if e else ("decoy" if nodes[cov].get("decoy") else "none"),
                "strength": nodes[int(tgt)]["strength"],
                "history_dependence": history_dependence(ep, int(tgt), cov),
            }
            if ep.roles[cov] == Role.KNOWN_FUTURE:
                row["lift"] = covariate_lift(ep, int(tgt), [cov])
            rows.append(row)
    return rows
