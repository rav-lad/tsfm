"""Leak-free per-channel normalization (reference §9.3).

Statistics are computed on the *visible history only*. Future labels are scaled
with the past statistics for the loss, but never contribute to them.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .episode import Episode, Role

MAD_TO_STD = 1.4826


@dataclass(frozen=True)
class RobustScaler:
    eps: float = 1e-6
    fallback_scale: float = 1.0
    method: str = "median_mad"  # or "mean_std"

    def stats(self, episode: Episode) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return ``(loc, scale, fallback_used)``, each of shape ``[C]``.

        Uses only positions that are observed *and* before the forecast origin.
        Known-future channels also use their history only (conservative policy).
        """
        L = episode.context_length
        hist = episode.values[:, :L].astype(np.float64)
        vis = episode.observed_mask[:, :L]
        C = episode.n_channels
        loc = np.zeros(C)
        scale = np.full(C, self.fallback_scale)
        fallback = np.zeros(C, dtype=bool)
        for c in range(C):
            x = hist[c, vis[c]]
            if x.size == 0:
                fallback[c] = True
                continue
            if self.method == "median_mad":
                m = float(np.median(x))
                s = MAD_TO_STD * float(np.median(np.abs(x - m)))
            elif self.method == "mean_std":
                m = float(x.mean())
                s = float(x.std())
            else:  # pragma: no cover
                raise ValueError(self.method)
            if s < self.eps:
                # MAD collapses on intermittent / piecewise-constant series:
                # fall back to the visible std before the constant fallback.
                s = float(x.std())
            if s < self.eps:
                s = self.fallback_scale
                fallback[c] = True
            loc[c], scale[c] = m, s
        return loc, scale, fallback

    def transform(self, episode: Episode) -> dict[str, np.ndarray]:
        """Normalized model input and normalized labels for one episode."""
        loc, scale, fallback = self.stats(episode)
        z = (episode.values.astype(np.float64) - loc[:, None]) / scale[:, None]
        L = episode.context_length
        model_input = np.where(episode.observed_mask, z, 0.0).astype(np.float32)
        target_future = z[:, L:].astype(np.float32)
        # Labels of non-target channels are never used: zero them so they cannot
        # be read by mistake.
        target_future[episode.roles != Role.TARGET] = 0.0
        return {
            "input_values": model_input,
            "input_mask": episode.observed_mask.copy(),
            "target_future": target_future,
            "loss_mask": episode.loss_mask.copy(),
            "scaling_stats": np.stack([loc, scale], axis=-1).astype(np.float32),
            "scale_fallback": fallback,
        }
