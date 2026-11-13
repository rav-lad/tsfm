"""Model V2: future patch tokens, horizons up to 1024 in one pass."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from tsfm.data import SyntheticEpisodeSource, collate_episodes, load_config
from tsfm.models import MODEL_PRESETS, V1_QUANTILES, PatchTransformerV0
from tsfm.training import split_batch, to_torch

CONFIGS = Path(__file__).parents[2] / "configs"
ATOL = 1e-5
Q = len(V1_QUANTILES)


@pytest.fixture(scope="module")
def model():
    torch.manual_seed(0)
    return PatchTransformerV0(MODEL_PRESETS["v2"]).eval()


@pytest.fixture(scope="module")
def source():
    return SyntheticEpisodeSource(load_config(CONFIGS / "train_v2.yaml"), split="debug")


def _batch(episodes):
    return to_torch(collate_episodes(episodes, pad_to_multiple=32))


@torch.no_grad()
def _run(model, batch, **kw):
    inputs, _ = split_batch(batch)
    return model(**inputs, **kw)


def test_parameter_count(model):
    assert 5.0e6 < model.num_parameters() < 6.0e6


@pytest.mark.parametrize("horizon", [1, 100, 1024, 1500])
def test_any_horizon(model, source, horizon):
    out = _run(model, _batch([source.episode(i) for i in range(4)]), horizon=horizon)
    assert out["quantiles"].shape[2:] == (horizon, Q)
    assert (out["quantiles"].diff(dim=-1) >= 0).all()


def test_one_pass_up_to_max_horizon(model, source):
    calls = []
    hook = model.blocks[0].register_forward_hook(lambda *a: calls.append(1))
    _run(model, _batch([source.episode(0)]), horizon=1024)
    assert len(calls) == 1
    _run(model, _batch([source.episode(0)]), horizon=1025)
    assert len(calls) == 3  # beyond max_horizon: rollout
    hook.remove()


def test_future_does_not_change_quantiles(model, source):
    batch = _batch([source.episode(i) for i in range(8)])
    ref = _run(model, batch)["quantiles"]
    origin = batch["input_values"].shape[-1] - batch["target_future"].shape[-1]
    noisy = dict(batch)
    noisy["input_values"] = batch["input_values"].clone()
    noisy["input_values"][..., origin:] = 1e3
    noisy["input_mask"] = batch["input_mask"].clone()
    noisy["input_mask"][..., origin:] = True
    assert torch.equal(_run(model, noisy)["quantiles"], ref)


def test_left_padding_does_not_change_quantiles(model, source):
    eps = [source.episode(i) for i in range(300)]
    short = next(e for e in eps if e.context_length == 128)
    long = next(e for e in eps if e.context_length == 2048)
    H = 96
    alone = _run(model, _batch([short]), horizon=H)["quantiles"]
    padded = _run(model, _batch([short, long]), horizon=H)["quantiles"][:1]
    torch.testing.assert_close(padded, alone, atol=ATOL, rtol=0)
