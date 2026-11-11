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


def augment_record(
    record: SyntheticRecord,
    rng: np.random.Generator,
    probability_any: float = 0.45,
    max_ops: int = 3,
    enabled: tuple[str, ...] | None = None,
) -> SyntheticRecord:
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
