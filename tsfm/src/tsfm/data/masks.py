"""Observation masks: role-based visibility and history missingness.

Four cases are kept distinct:
1. observed zero      -> value 0, observed_mask 1
2. missing value      -> observed_mask 0, not supervised
3. hidden target future -> observed_mask 0, loss_mask 1
4. padding            -> handled at collate time (padding_mask)
"""

from __future__ import annotations

from typing import Any

import numpy as np

from .episode import Role


def role_visibility(roles: np.ndarray, L: int, H: int) -> np.ndarray:
    """bool [C, L+H]: history visible for all, future only for KNOWN_FUTURE."""
    C = len(roles)
    mask = np.zeros((C, L + H), dtype=bool)
    mask[:, :L] = True
    mask[roles == Role.KNOWN_FUTURE, L:] = True
    return mask


def sample_history_missingness(
    rng: np.random.Generator,
    L: int,
    probs: dict[str, float],
    sparse_rate: tuple[float, float] = (0.01, 0.10),
    block_frac: tuple[float, float] = (0.05, 0.30),
    min_observed: int = 32,
    keep_last: int = 1,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Missingness pattern for the history of one channel.

    Returns ``(visible[L], info)``. Guarantees at least ``min_observed`` visible
    points and keeps the last ``keep_last`` points visible, so every episode has a
    usable anchor before the origin.
    """
    kinds = list(probs)
    p = np.array([probs[k] for k in kinds], dtype=float)
    kind = kinds[int(rng.choice(len(kinds), p=p / p.sum()))]
    visible = np.ones(L, dtype=bool)
    info: dict[str, Any] = {"kind": kind}
    protected_end = L - keep_last

    if kind == "sparse":
        rate = float(rng.uniform(*sparse_rate))
        drop = rng.random(protected_end) < rate
        visible[:protected_end] &= ~drop
        info["rate"] = rate
    elif kind == "block":
        frac = float(rng.uniform(*block_frac))
        size = max(1, int(round(frac * L)))
        size = min(size, max(0, L - min_observed - keep_last))
        if size > 0:
            start = int(rng.integers(0, protected_end - size + 1))
            visible[start : start + size] = False
            info.update(start=start, size=size)
    elif kind != "complete":
        raise ValueError(f"unknown missingness kind {kind!r}")

    if visible.sum() < min_observed:  # restore the most recent points
        need = min_observed - int(visible.sum())
        hidden = np.flatnonzero(~visible)[::-1][:need]
        visible[hidden] = True
        info["restored"] = int(len(hidden))
    info["observed_fraction"] = float(visible.mean())
    return visible, info
