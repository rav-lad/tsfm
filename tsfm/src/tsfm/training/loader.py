"""PyTorch bridge for the numpy data pipeline (roadmap phase 3, reference §10.2).

Workers build whole *batches* (generate + collate) so that only a handful of
numpy arrays cross the process boundary, not hundreds of ``Episode`` objects.

Determinism: batch ``k`` always holds sample ids ``[k*B, (k+1)*B)``. Worker ``w``
of ``n`` builds batches ``k = w, w+n, w+2n, ...`` and the DataLoader yields them
round-robin, so the stream of batches is identical for any ``num_workers``.
Resuming a run is ``start_batch = steps_done``.

``target_future`` and ``loss_mask`` are labels: ``split_batch`` separates them
so that ``model.forward`` can be called on the inputs alone.
"""

from __future__ import annotations

import math
import os
from pathlib import Path
from typing import Any, Iterator

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, IterableDataset, get_worker_info

from tsfm.data import SyntheticEpisodeSource, collate_episodes, load_config
from tsfm.data.frozen import FrozenEpisodeSet

# loss_mask is a label too: it reveals which future target points are missing.
LABEL_KEYS = ("target_future", "loss_mask")

NumpyBatch = dict[str, np.ndarray]
TorchBatch = dict[str, torch.Tensor]


def _as_config(config: dict[str, Any] | str | Path) -> dict[str, Any]:
    return config if isinstance(config, dict) else load_config(config)


class SyntheticBatchDataset(IterableDataset):
    """Infinite (or ``num_batches``-long) stream of on-the-fly collated batches."""

    def __init__(
        self,
        config: dict[str, Any] | str | Path,
        split: str = "train",
        batch_size: int = 64,
        start_batch: int = 0,
        num_batches: int | None = None,
        pad_to_multiple: int = 32,
        seed: int | None = None,
    ) -> None:
        self.config = _as_config(config)
        self.split = split
        self.batch_size = int(batch_size)
        self.start_batch = int(start_batch)
        self.num_batches = num_batches
        self.pad_to_multiple = int(pad_to_multiple)
        self.seed = seed
        self._source: SyntheticEpisodeSource | None = None

    def __getstate__(self) -> dict[str, Any]:
        # each worker rebuilds its generators instead of unpickling them
        return {**self.__dict__, "_source": None}

    @property
    def source(self) -> SyntheticEpisodeSource:
        if self._source is None:
            self._source = SyntheticEpisodeSource(self.config, split=self.split, seed=self.seed)
        return self._source

    def sample_ids(self, k: int) -> range:
        return range(k * self.batch_size, (k + 1) * self.batch_size)

    def batch(self, k: int) -> NumpyBatch:
        episodes = [self.source.episode(i) for i in self.sample_ids(k)]
        return collate_episodes(episodes, pad_to_multiple=self.pad_to_multiple)

    def __iter__(self) -> Iterator[NumpyBatch]:
        info = get_worker_info()
        wid, nw = (info.id, info.num_workers) if info is not None else (0, 1)
        end = self.start_batch + self.num_batches if self.num_batches is not None else None
        k = self.start_batch + wid
        while end is None or k < end:
            yield self.batch(k)
            k += nw


class FrozenBatchDataset(Dataset):
    """Fixed batches over a frozen suite: batch ``k`` = episodes ``[k*B, (k+1)*B)``."""

    def __init__(self, suite_dir: str | Path, batch_size: int = 64, pad_to_multiple: int = 32) -> None:
        self.suite_dir = Path(suite_dir)
        self.batch_size = int(batch_size)
        self.pad_to_multiple = int(pad_to_multiple)
        # verify hashes once here; workers reload without re-hashing
        self._suite: FrozenEpisodeSet | None = FrozenEpisodeSet(self.suite_dir, verify=True)
        self._n = len(self._suite)

    def __getstate__(self) -> dict[str, Any]:
        return {**self.__dict__, "_suite": None}

    @property
    def suite(self) -> FrozenEpisodeSet:
        if self._suite is None:
            self._suite = FrozenEpisodeSet(self.suite_dir, verify=False)
        return self._suite

    def __len__(self) -> int:
        return math.ceil(self._n / self.batch_size)

    def __getitem__(self, k: int) -> NumpyBatch:
        if not 0 <= k < len(self):
            raise IndexError(k)
        ids = range(k * self.batch_size, min((k + 1) * self.batch_size, self._n))
        return collate_episodes([self.suite[i] for i in ids], pad_to_multiple=self.pad_to_multiple)


_BLAS_THREAD_VARS = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")


def _single_thread_worker(_: int) -> None:
    torch.set_num_threads(1)


def _loader(dataset: Dataset, num_workers: int, pin_memory: bool | None, prefetch_factor: int) -> DataLoader:
    if pin_memory is None:
        pin_memory = torch.cuda.is_available()
    extra: dict[str, Any] = {}
    if num_workers > 0:
        # one BLAS thread per worker: N workers x M threads oversubscribes the CPU.
        # Spawned workers read these at numpy import; explicit user settings win.
        for var in _BLAS_THREAD_VARS:
            os.environ.setdefault(var, "1")
        extra = {
            "prefetch_factor": prefetch_factor,
            "persistent_workers": True,
            "worker_init_fn": _single_thread_worker,
        }
    return DataLoader(
        dataset,
        batch_size=None,  # datasets already yield collated batches
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=_numpy_to_torch,
        **extra,
    )


def make_train_loader(
    config: dict[str, Any] | str | Path,
    batch_size: int = 64,
    num_workers: int = 4,
    start_batch: int = 0,
    num_batches: int | None = None,
    split: str = "train",
    pad_to_multiple: int = 32,
    pin_memory: bool | None = None,
    prefetch_factor: int = 4,
) -> DataLoader:
    ds = SyntheticBatchDataset(config, split, batch_size, start_batch, num_batches, pad_to_multiple)
    return _loader(ds, num_workers, pin_memory, prefetch_factor)


def make_frozen_loader(
    suite_dir: str | Path,
    batch_size: int = 64,
    num_workers: int = 0,
    pad_to_multiple: int = 32,
    pin_memory: bool | None = None,
    prefetch_factor: int = 2,
) -> DataLoader:
    ds = FrozenBatchDataset(suite_dir, batch_size, pad_to_multiple)
    return _loader(ds, num_workers, pin_memory, prefetch_factor)


def _numpy_to_torch(batch: NumpyBatch) -> TorchBatch:
    return {k: torch.from_numpy(np.ascontiguousarray(v)) for k, v in batch.items()}


def to_torch(batch: NumpyBatch | TorchBatch, device: str | torch.device = "cpu") -> TorchBatch:
    """Convert (if needed) and move a batch to ``device``."""
    non_blocking = torch.device(device).type == "cuda"
    out = {}
    for k, v in batch.items():
        t = v if isinstance(v, torch.Tensor) else torch.from_numpy(np.ascontiguousarray(v))
        out[k] = t.to(device, non_blocking=non_blocking)
    return out


def split_batch(batch: TorchBatch) -> tuple[TorchBatch, TorchBatch]:
    """``(inputs, labels)``: only ``inputs`` may be passed to ``model.forward``."""
    inputs = {k: v for k, v in batch.items() if k not in LABEL_KEYS}
    labels = {k: v for k, v in batch.items() if k in LABEL_KEYS}
    return inputs, labels
