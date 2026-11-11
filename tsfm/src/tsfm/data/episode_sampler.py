"""Cut a forecasting episode out of a source record (reference §3.1, §8.3)."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .episode import Episode, Role, SyntheticRecord
from .masks import role_visibility, sample_history_missingness


@dataclass
class EpisodeSampler:
    missingness: dict[str, float] = field(default_factory=lambda: {"complete": 0.70, "sparse": 0.20, "block": 0.10})
    min_history_observed: int = 32
    buffer: int = 0  # initial steps reserved for causal lags (multivariate)

    def required_length(self, L: int, H: int, slack: int = 0) -> int:
        return self.buffer + L + H + slack

    def sample(
        self,
        rng: np.random.Generator,
        rng_missing: np.random.Generator,
        record: SyntheticRecord,
        context_length: int,
        prediction_length: int,
        roles: np.ndarray | None = None,
    ) -> Episode:
        L, H = int(context_length), int(prediction_length)
        T, C = record.values.shape
        lo = self.buffer + L
        hi = T - H
        if hi < lo:
            raise ValueError(f"record too short: T={T}, need >= {self.buffer + L + H}")
        origin = int(rng.integers(lo, hi + 1))
        values = record.values[origin - L : origin + H].T.astype(np.float32)  # [C, L+H]

        if roles is None:
            roles = np.full(C, Role.TARGET, dtype=np.int8)
        roles = np.asarray(roles, dtype=np.int8)
        if len(roles) != C:
            raise ValueError("one role per channel required")

        observed = role_visibility(roles, L, H)
        miss_info = []
        for c in range(C):
            vis, info = sample_history_missingness(
                rng_missing, L, self.missingness, min_observed=min(self.min_history_observed, L)
            )
            observed[c, :L] &= vis
            miss_info.append(info)

        loss_mask = np.zeros((C, H), dtype=bool)
        loss_mask[roles == Role.TARGET] = True
        # Non-finite future values (should not happen) are excluded from the loss.
        loss_mask &= np.isfinite(values[:, L:])

        meta = dict(record.metadata)
        meta["missingness"] = miss_info
        return Episode(
            values=values,
            observed_mask=observed,
            loss_mask=loss_mask,
            roles=roles,
            context_length=L,
            prediction_length=H,
            origin=origin,
            meta=meta,
        )
