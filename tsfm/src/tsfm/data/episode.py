"""Canonical data objects: source records and forecasting episodes.

Vocabulary (reference §3):
- *record*  : aligned multivariate array produced by a generator, shape ``[T, C]``.
- *episode* : what the network sees, shape ``[C, L + H]`` with roles and masks.

``Episode.values`` holds the future labels and must stay on the dataset/trainer
side. The model only ever receives ``Episode.model_input()`` (or the collated
``input_values``), in which every non-observed position is zeroed.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import IntEnum
from typing import Any

import numpy as np


class Role(IntEnum):
    TARGET = 0  # history visible, future hidden and supervised
    PAST_ONLY = 1  # history visible, future hidden, never supervised
    KNOWN_FUTURE = 2  # history and future visible (known at the cutoff)


class Support:
    """Value support of a record (string constants), used to restrict augmentations."""

    REAL = "real"
    NONNEGATIVE = "nonnegative"
    INTEGER = "integer"  # non-negative integers (counts)


SUPPORTS = (Support.REAL, Support.NONNEGATIVE, Support.INTEGER)


@dataclass
class SyntheticRecord:
    values: np.ndarray  # [T, C], float64 during generation
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.values.ndim == 1:
            self.values = self.values[:, None]
        if self.values.ndim != 2:
            raise ValueError(f"record values must be [T, C], got shape {self.values.shape}")

    @property
    def length(self) -> int:
        return int(self.values.shape[0])

    @property
    def n_channels(self) -> int:
        return int(self.values.shape[1])


@dataclass
class Episode:
    values: np.ndarray  # float32 [C, L+H]  -- contains labels, trainer only
    observed_mask: np.ndarray  # bool [C, L+H]
    loss_mask: np.ndarray  # bool [C, H]
    roles: np.ndarray  # int8 [C]
    context_length: int
    prediction_length: int
    origin: int  # index of the first future step in the source record
    meta: dict[str, Any] = field(default_factory=dict)

    @property
    def n_channels(self) -> int:
        return int(self.values.shape[0])

    @property
    def target_channels(self) -> np.ndarray:
        return np.flatnonzero(self.roles == Role.TARGET)

    def history(self) -> np.ndarray:
        return self.values[:, : self.context_length]

    def future(self) -> np.ndarray:
        return self.values[:, self.context_length :]

    def model_input(self) -> np.ndarray:
        """The only view of ``values`` the model may receive (raw scale)."""
        return np.where(self.observed_mask, self.values, 0.0).astype(np.float32)

    def n_supervised_positions(self) -> int:
        return int(self.loss_mask.sum())
