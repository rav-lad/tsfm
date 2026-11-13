"""Support-aware augmentations on source records (reference §9.1).

Each episode samples ``none / one / multiple`` operations; every applied
operation is logged in metadata. Operations act jointly on all channels of a
record so cross-series relations are preserved (needed for the multivariate
stage).

Support rules:
- real        : scale (pos.), offset, sign flip, noise, outliers
- nonnegative : positive scale, positive offset, noise clipped at 0
- integer     : no value-changing op (would break count semantics); outliers as
                integer bursts only
"""

from __future__ import annotations

from typing import Any

import numpy as np

from .episode import Support, SyntheticRecord
from .generators.base import log_uniform

ALLOWED = {
    Support.REAL: ("scale", "offset", "sign_flip", "noise", "outliers"),
    Support.NONNEGATIVE: ("scale", "offset", "noise", "outliers"),
    Support.INTEGER: ("outliers",),
}


def _apply(op: str, x: np.ndarray, support: str, rng: np.random.Generator) -> tuple[np.ndarray, dict[str, Any]]:
    std = float(np.std(x)) or 1.0
    if op == "scale":
        s = log_uniform(rng, 1e-2, 1e2)
        return x * s, {"op": op, "factor": s}
    if op == "offset":
        o = float(rng.normal(0, 3)) * std
        if support == Support.NONNEGATIVE:
            o = abs(o)
        return x + o, {"op": op, "offset": o}
    if op == "sign_flip":
        return -x, {"op": op}
    if op == "noise":
        r = float(rng.uniform(0.05, 0.3))
        y = x + r * std * rng.standard_normal(x.shape)
        if support == Support.NONNEGATIVE:
            y = np.maximum(y, 0.0)
        return y, {"op": op, "relative_std": r}
    if op == "outliers":
        rate = log_uniform(rng, 1e-3, 1e-2)
        hit = rng.random(x.shape[0]) < rate
        size = std * rng.uniform(3, 8, x.shape[0])
        if support == Support.INTEGER:
            bump = np.round(size)
        else:
            bump = size * rng.choice([-1.0, 1.0], x.shape[0])
            if support == Support.NONNEGATIVE:
                bump = np.abs(bump)
        y = x.copy()
        y[hit] += bump[hit, None] if x.ndim == 2 else bump[hit]
        return y, {"op": op, "rate": rate, "n": int(hit.sum())}
    raise ValueError(op)  # pragma: no cover


# ---------------------------------------------------------------------- TiRex augmentations
# Independent of the ops above, each with its own probability (TiRex, Auer et al.
# 2025, Sec. 3 / App. B). They act on the time axis of a [T, C] record.


def _amplitude_modulation(x: np.ndarray, support: str, rng: np.random.Generator) -> tuple[np.ndarray, dict[str, Any]]:
    """x * a_t, a_t piecewise linear through k ~ U{0..5} change points, amplitudes ~ N(1, 1).
    Real support: applied to the standardized series so the level is kept."""
    T = x.shape[0]
    k = int(rng.integers(0, 6))
    knots_t = np.r_[0, np.sort(rng.integers(1, T - 1, k)) if T > 2 else [], T - 1].astype(float)
    knots_a = rng.normal(1.0, 1.0, len(knots_t))
    if support != Support.REAL:
        knots_a = np.abs(knots_a)
    a = np.interp(np.arange(T), knots_t, knots_a)[:, None]
    if support == Support.REAL:
        mu, sd = x.mean(axis=0), x.std(axis=0)
        sd = np.where(sd > 1e-12, sd, 1.0)
        return mu + sd * ((x - mu) / sd) * a, {"op": "amplitude_modulation", "n_knots": k}
    y = x * a
    return (np.round(y) if support == Support.INTEGER else y), {"op": "amplitude_modulation", "n_knots": k}


def _censor(x: np.ndarray, support: str, rng: np.random.Generator) -> tuple[np.ndarray, dict[str, Any]]:
    """Clip from below or above at an empirical quantile q ~ U(0, 1)."""
    q = float(rng.uniform(0, 1))
    thr = np.quantile(x, q, axis=0, method="lower")
    upper = bool(rng.random() < 0.5)
    return (np.minimum(x, thr) if upper else np.maximum(x, thr)), {"op": "censor", "quantile": q, "upper": upper}


def _spike_injection(x: np.ndarray, support: str, rng: np.random.Generator) -> tuple[np.ndarray, dict[str, Any]]:
    """Periodic spikes: period ~ U(10, min(512, T)), tophat / RBF / linear kernel,
    width in [0.05, 0.2] x period, height in [0.5, 3] x std (one sign per record)."""
    T = x.shape[0]
    period = float(rng.uniform(10, max(11, min(512, T))))
    width = max(1.0, float(rng.uniform(0.05, 0.2)) * period)
    kernel = str(rng.choice(["tophat", "rbf", "linear"]))
    height = float(rng.uniform(0.5, 3.0)) * (x.std(axis=0) + 1e-12)
    sign = 1.0 if support != Support.REAL or rng.random() < 0.5 else -1.0
    d = (np.arange(T) - rng.uniform(0, period)) % period
    d = np.minimum(d, period - d) / width  # distance to the nearest spike centre, in widths
    if kernel == "tophat":
        shape = (d <= 0.5).astype(float)
    elif kernel == "rbf":
        shape = np.exp(-0.5 * (2 * d) ** 2)
    else:
        shape = np.clip(1 - d, 0, None)
    bump = sign * shape[:, None] * height
    if support == Support.INTEGER:
        bump = np.round(bump)
    return x + bump, {"op": "spike_injection", "period": period, "width": width, "kernel": kernel}


TIREX_OPS = {"amplitude_modulation": _amplitude_modulation, "censor": _censor, "spike_injection": _spike_injection}


def augment_record(
    record: SyntheticRecord,
    rng: np.random.Generator,
    probability_any: float = 0.45,
    max_ops: int = 3,
    enabled: tuple[str, ...] | None = None,
    tirex: dict[str, float] | None = None,
) -> SyntheticRecord:
    """``enabled`` restricts the ops above; ``tirex`` maps TiRex ops to their own
    probabilities (univariate records only, drawn after the other ops so configs
    without it keep their exact random stream)."""
    if "roles" in record.metadata:
        return _augment_multivariate(record, rng, probability_any, max_ops, enabled)
    support = record.metadata.get("support", Support.REAL)
    ops_ok = [o for o in ALLOWED[support] if enabled is None or o in enabled]
    applied: list[dict[str, Any]] = []
    x = record.values
    if ops_ok and rng.random() < probability_any:
        n_ops = int(rng.integers(1, min(max_ops, len(ops_ok)) + 1))
        for op in rng.choice(ops_ok, size=n_ops, replace=False):
            x, info = _apply(str(op), x, support, rng)
            applied.append(info)
    for op, p in (tirex or {}).items():
        if rng.random() < p:
            x, info = TIREX_OPS[op](x, support, rng)
            applied.append(info)
    meta = dict(record.metadata)
    meta["augmentations"] = applied
    return SyntheticRecord(values=x, metadata=meta)


def _augment_multivariate(
    record: SyntheticRecord,
    rng: np.random.Generator,
    probability_any: float,
    max_ops: int,
    enabled: tuple[str, ...] | None,
) -> SyntheticRecord:
    """Per-channel augmentation for relational records.

    Value transforms (scale, offset, sign, noise, outliers) applied *per channel*
    keep cross-channel dependence intact (an affine map of a driver is still a
    driver). KNOWN_FUTURE channels (calendars, plans) are left untouched: noise
    on a planned promotion calendar would make it no longer "known". Time
    transforms that would break alignment are not used here.
    """
    from .episode import Role

    roles = record.metadata["roles"]
    x = record.values.copy()
    applied: list[dict[str, Any]] = []
    ops_all = [o for o in ALLOWED[Support.REAL] if enabled is None or o in enabled]
    if ops_all and rng.random() < probability_any:
        for c, role in enumerate(roles):
            if role == Role.KNOWN_FUTURE or rng.random() < 0.5:
                continue
            n_ops = int(rng.integers(1, min(max_ops, len(ops_all)) + 1))
            for op in rng.choice(ops_all, size=n_ops, replace=False):
                col, info = _apply(str(op), x[:, c], Support.REAL, rng)
                x[:, c] = col
                applied.append({"channel": c, **info})
    meta = dict(record.metadata)
    meta["augmentations"] = applied
    return SyntheticRecord(values=x, metadata=meta)
