"""On-the-fly synthetic episode source (reference §10.3).

``SyntheticEpisodeSource(config, split).episode(sample_id)`` is a pure function
of ``(seed, split, sample_id, config[, progress])``: it can be called from any
worker in any order, and replayed from metadata alone.

Pipeline per sample::

    uni / multi <- FAMILY stream (curriculum probability, §15.3)
    family      <- FAMILY stream
    L, H        <- EPISODE stream
    record      <- generator(RECORD stream, length = L + H + slack)
    record      <- augment(AUGMENT stream)
    episode     <- cut + roles + masks (EPISODE, MISSING streams)
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Iterator

import numpy as np

from .augmentations import augment_record
from .config import load_config  # re-exported
from .episode import Episode
from .episode_sampler import EpisodeSampler
from .generators import REGISTRY
from .multivariate import MultivariateGenerator
from .rng import Stream, sample_rng, sample_seed
from .validation import ContractError, check_episode, check_record

MAX_ATTEMPTS = 5
MULTIVARIATE = MultivariateGenerator.family

__all__ = ["SyntheticEpisodeSource", "ExposureTracker", "effective_mix", "load_config", "curriculum_probability"]


def effective_mix(mix: dict[str, float]) -> tuple[dict[str, float], list[str]]:
    """Keep implemented families only and renormalize. Returns (mix, dropped)."""
    kept = {k: float(v) for k, v in mix.items() if k in REGISTRY and v > 0}
    dropped = [k for k, v in mix.items() if k not in REGISTRY and v > 0]
    total = sum(kept.values())
    if total <= 0:
        raise ValueError("no implemented generator in generator_mix")
    return {k: v / total for k, v in kept.items()}, dropped


def curriculum_probability(milestones: list[dict[str, float]], progress: float) -> float:
    """Piecewise-linear interpolation of ``multivariate_probability`` over progress."""
    pts = sorted((float(m["progress"]), float(m["multivariate_probability"])) for m in milestones)
    xs, ys = zip(*pts)
    return float(np.interp(np.clip(progress, 0.0, 1.0), xs, ys))


def _dist(d: dict) -> tuple[np.ndarray, np.ndarray]:
    keys = np.array([int(k) for k in d])
    p = np.array([float(v) for v in d.values()])
    return keys, p / p.sum()


class SyntheticEpisodeSource:
    def __init__(self, config: dict[str, Any], split: str = "train", seed: int | None = None) -> None:
        self.config = config
        self.split = split
        self.seed = int(seed if seed is not None else config["seed"])
        self.mix, self.dropped_families = effective_mix(config["generator_mix"])
        self._families = list(self.mix)
        self._family_p = np.array([self.mix[f] for f in self._families])
        gen_cfg = config.get("univariate_generators", {}) or {}
        prior = config.get("prior", {}) or {}  # shared holdout hooks
        def params(f: str) -> dict[str, Any]:
            return {**prior, **(gen_cfg.get(f) or {})}

        self.generators = {f: REGISTRY[f](**params(f)) for f in self._families if f != "composite"}
        if "composite" in self._families:
            # G7 composes the *configured* base generators, so holdout exclusions
            # (held-out kinds, period band) also hold inside composites.
            comp_cfg = params("composite")
            base = self.generators or {f: REGISTRY[f](**prior) for f in REGISTRY if f != "composite"}
            self.generators["composite"] = REGISTRY["composite"](
                components=base, component_mix=comp_cfg.pop("component_mix", None), **comp_cfg
            )

        mv = config.get("multivariate", {}) or {}
        self.mv_enabled = bool(mv.get("enabled", False))
        self.mv_generator = None
        if self.mv_enabled:
            mv_params = {k: v for k, v in mv.items() if k != "enabled"}
            self.mv_generator = MultivariateGenerator(self.generators, self.mix, **mv_params)
        cur = config.get("curriculum", {}) or {}
        self.milestones = cur.get("milestones")
        self.total_episodes = cur.get("total_episodes")
        self.fixed_mv_probability = float(mv.get("probability", 0.0))

        seq = config["sequence"]
        self._L, self._pL = _dist(seq["context_lengths"])
        self._H, self._pH = _dist(seq["horizons"])
        self.slack_fraction = float(seq.get("origin_slack_fraction", 0.25))
        self.sampler = EpisodeSampler(
            missingness=config.get("missingness", {"complete": 1.0}),
            min_history_observed=int(seq.get("min_history_observed", 32)),
        )
        aug = config.get("augmentations", {}) or {}
        self.aug_probability = float(aug.get("probability_any", 0.0))
        self.aug_max_ops = int(aug.get("max_ops", 3))
        self.aug_ops = tuple(aug["ops"]) if aug.get("ops") is not None else None
        self.aug_tirex = dict(aug.get("tirex") or {})

    # ------------------------------------------------------------------ curriculum
    def multivariate_probability(self, sample_id: int, progress: float | None = None) -> float:
        if not self.mv_enabled:
            return 0.0
        if self.milestones:
            if progress is None:
                if not self.total_episodes:
                    raise ValueError("curriculum needs `progress` or curriculum.total_episodes")
                progress = sample_id / float(self.total_episodes)
            return curriculum_probability(self.milestones, progress)
        return self.fixed_mv_probability

    # ------------------------------------------------------------------ core
    def _rng(self, sample_id: int, stream: Stream, attempt: int) -> np.random.Generator:
        # attempt > 0 only after an implementation anomaly; shifts every stream
        return sample_rng(self.seed, self.split, sample_id, int(stream) + 16 * attempt)

    def episode(self, sample_id: int, family: str | None = None, progress: float | None = None) -> Episode:
        last_err: Exception | None = None
        for attempt in range(MAX_ATTEMPTS):
            try:
                return self._episode(sample_id, attempt, family, progress)
            except ContractError as err:
                last_err = err
        raise RuntimeError(f"sample {sample_id}: {MAX_ATTEMPTS} invalid attempts") from last_err

    def _episode(self, sample_id: int, attempt: int, family: str | None, progress: float | None) -> Episode:
        r_fam = self._rng(sample_id, Stream.FAMILY, attempt)
        r_ep = self._rng(sample_id, Stream.EPISODE, attempt)
        u_mv = r_fam.random()  # always drawn, so the stream layout never changes
        if family is None:
            if u_mv < self.multivariate_probability(sample_id, progress):
                family = MULTIVARIATE
            else:
                family = self._families[int(r_fam.choice(len(self._families), p=self._family_p))]
        L = int(r_ep.choice(self._L, p=self._pL))
        H = int(r_ep.choice(self._H, p=self._pH))
        slack = int(r_ep.integers(0, int(self.slack_fraction * (L + H)) + 1))
        length = self.sampler.required_length(L, H, slack)

        gen = self.mv_generator if family == MULTIVARIATE else self.generators[family]
        if gen is None:
            raise ValueError("multivariate requested but multivariate.enabled is false")
        record = gen.sample(self._rng(sample_id, Stream.RECORD, attempt), length, L)
        check_record(record)
        record = augment_record(
            record, self._rng(sample_id, Stream.AUGMENT, attempt), self.aug_probability, self.aug_max_ops,
            enabled=self.aug_ops, tirex=self.aug_tirex,
        )
        check_record(record)
        roles = record.metadata.get("roles")
        ep = self.sampler.sample(
            r_ep, self._rng(sample_id, Stream.MISSING, attempt), record, L, H,
            roles=np.array(roles, np.int8) if roles is not None else None,
        )
        ep.meta.update(
            split=self.split,
            master_seed=self.seed,
            sample_id=int(sample_id),
            attempt=attempt,
            record_seed=sample_seed(self.seed, self.split, sample_id, int(Stream.RECORD) + 16 * attempt),
            record_length=length,
        )
        check_episode(ep)
        return ep

    def __iter__(self) -> Iterator[Episode]:
        i = 0
        while True:
            yield self.episode(i)
            i += 1

    def episodes(self, start: int, n: int) -> Iterator[Episode]:
        for i in range(start, start + n):
            yield self.episode(i)


@dataclass
class ExposureTracker:
    """The three quantity metrics of reference §7.2 (plus a few extras)."""

    n_source_draws: int = 0
    n_episodes_seen: int = 0
    n_supervised_scalar_positions: int = 0
    n_input_scalar_positions: int = 0
    family_counts: Counter = field(default_factory=Counter)
    channel_counts: Counter = field(default_factory=Counter)
    _sources: set = field(default_factory=set)

    def update(self, ep: Episode) -> None:
        self.n_episodes_seen += 1
        key = (ep.meta.get("split"), ep.meta.get("sample_id"), ep.meta.get("attempt"))
        if key not in self._sources:
            self._sources.add(key)
            self.n_source_draws += 1
        self.n_supervised_scalar_positions += ep.n_supervised_positions()
        self.n_input_scalar_positions += int(ep.observed_mask.sum())
        self.family_counts[ep.meta.get("family")] += 1
        self.channel_counts[ep.n_channels] += 1

    def summary(self) -> dict[str, Any]:
        n = max(self.n_episodes_seen, 1)
        return {
            "n_source_draws": self.n_source_draws,
            "n_episodes_seen": self.n_episodes_seen,
            "n_supervised_scalar_positions": self.n_supervised_scalar_positions,
            "n_input_scalar_positions": self.n_input_scalar_positions,
            "family_fraction": {k: v / n for k, v in sorted(self.family_counts.items())},
            "channels_fraction": {k: v / n for k, v in sorted(self.channel_counts.items())},
        }
