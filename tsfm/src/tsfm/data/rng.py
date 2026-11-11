"""Deterministic, per-sample random number streams.

Every sample is identified by ``(master_seed, split, sample_id)``. Its RNG never
depends on worker ids, batch composition or iteration order, so changing
``num_workers`` or the batch size does not change the data (reference §10.4).

Each sample owns several independent *streams* so that, for example, changing the
episode sampler (L, H, missingness) does not change the underlying source record.
"""

from __future__ import annotations

import zlib
from enum import IntEnum

import numpy as np

SPLITS: dict[str, int] = {
    "train": 0,
    "val_id": 1,
    "val_ood": 2,
    "test_syn": 3,
    "debug": 9,
}


class Stream(IntEnum):
    """Independent sub-streams of one sample."""

    FAMILY = 0  # which generator family
    RECORD = 1  # the generative mechanism and its noise
    EPISODE = 2  # L, H, origin, roles
    AUGMENT = 3  # augmentations
    MISSING = 4  # missingness pattern


def split_code(split: str) -> int:
    """Integer namespace of a split.

    Base splits have fixed codes. Named sub-suites such as ``"val_ood/noise"``
    get a stable code derived from their name (crc32), offset so it never
    collides with a base split; each suite therefore draws independent sources.
    """
    if split in SPLITS:
        return SPLITS[split]
    if "/" not in split:
        raise ValueError(f"unknown split {split!r}: use one of {sorted(SPLITS)} or '<base>/<suite>'")
    base = split.split("/", 1)[0]
    if base not in SPLITS:
        raise ValueError(f"unknown base split in {split!r}")
    return 1000 + zlib.crc32(split.encode("utf-8"))


def sample_rng(master_seed: int, split: str, sample_id: int, stream: Stream | int) -> np.random.Generator:
    """RNG for one stream of one sample. Pure function of its arguments."""
    if sample_id < 0:
        raise ValueError("sample_id must be >= 0")
    seq = np.random.SeedSequence(
        entropy=int(master_seed),
        spawn_key=(split_code(split), int(sample_id), int(stream)),
    )
    return np.random.default_rng(seq)


def sample_seed(master_seed: int, split: str, sample_id: int, stream: Stream | int) -> int:
    """A 64-bit integer seed for the same stream (stored in metadata for replay)."""
    seq = np.random.SeedSequence(
        entropy=int(master_seed),
        spawn_key=(split_code(split), int(sample_id), int(stream)),
    )
    return int(seq.generate_state(2, dtype=np.uint32).view(np.uint64)[0])
