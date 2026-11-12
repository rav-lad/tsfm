"""PyTorch loader: determinism across workers, resume, label separation."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from tsfm.data import SyntheticEpisodeSource, collate_episodes, load_config
from tsfm.data.frozen import write_suite
from tsfm.training import (
    LABEL_KEYS,
    SyntheticBatchDataset,
    make_frozen_loader,
    make_train_loader,
    split_batch,
    to_torch,
)

CONFIG = Path(__file__).parents[2] / "configs" / "data_v1.yaml"
B = 8


@pytest.fixture(scope="module")
def config():
    return load_config(CONFIG)


def _take(loader, n):
    out = []
    for batch in loader:
        out.append(batch)
        if len(out) == n:
            break
    return out


def _assert_same(a, b):
    assert a.keys() == b.keys()
    for k in a:
        assert torch.equal(a[k], b[k]), k


def test_batch_k_is_collate_of_sample_ids(config):
    ds = SyntheticBatchDataset(config, batch_size=B)
    src = SyntheticEpisodeSource(config, split="train")
    expected = collate_episodes([src.episode(i) for i in range(3 * B, 4 * B)], pad_to_multiple=32)
    got = ds.batch(3)
    for k in expected:
        np.testing.assert_array_equal(got[k], expected[k])


def test_stream_is_independent_of_num_workers(config):
    ref = _take(make_train_loader(config, batch_size=B, num_workers=0, num_batches=4), 4)
    par = _take(make_train_loader(config, batch_size=B, num_workers=2, num_batches=4), 4)
    assert len(ref) == len(par) == 4
    for a, b in zip(ref, par):
        _assert_same(a, b)


def test_resume_from_start_batch(config):
    full = _take(make_train_loader(config, batch_size=B, num_workers=0, num_batches=5), 5)
    resumed = _take(make_train_loader(config, batch_size=B, num_workers=0, start_batch=2, num_batches=3), 3)
    for a, b in zip(full[2:], resumed):
        _assert_same(a, b)


def test_tensors_shapes_and_dtypes(config):
    (batch,) = _take(make_train_loader(config, batch_size=B, num_workers=0, num_batches=1), 1)
    Bx, C, W = batch["input_values"].shape
    assert Bx == B and W % 32 == 0
    assert batch["input_values"].dtype == torch.float32
    assert batch["input_mask"].dtype == torch.bool
    assert batch["target_future"].shape[:2] == (B, C)
    batch = to_torch(batch, "cpu")
    assert batch["input_values"].device.type == "cpu"


def test_split_batch_keeps_labels_out_of_inputs(config):
    (batch,) = _take(make_train_loader(config, batch_size=B, num_workers=0, num_batches=1), 1)
    inputs, labels = split_batch(batch)
    assert set(labels) == set(LABEL_KEYS)
    assert not set(inputs) & set(LABEL_KEYS)
    assert set(inputs) | set(labels) == set(batch)


def test_frozen_loader_roundtrip(config, tmp_path):
    src = SyntheticEpisodeSource(config, split="val_id")
    episodes = [src.episode(i) for i in range(2 * B + 3)]
    write_suite(tmp_path / "suite", episodes, {"name": "test"})
    batches = list(make_frozen_loader(tmp_path / "suite", batch_size=B))
    assert [b["input_values"].shape[0] for b in batches] == [B, B, 3]
    expected = collate_episodes(episodes[:B], pad_to_multiple=32)
    for k in expected:
        np.testing.assert_array_equal(batches[0][k].numpy(), expected[k])
