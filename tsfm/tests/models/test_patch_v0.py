"""Model V0 guard rails: size, no leakage, padding / missingness invariance, overfit."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from tsfm.data import Role, SyntheticEpisodeSource, collate_episodes, load_config
from tsfm.models import PatchTransformerV0, PatchV0Config
from tsfm.training import split_batch, to_torch
from tsfm.training.losses import denormalize, masked_mae

CONFIGS = Path(__file__).parents[2] / "configs"
ATOL = 1e-5


@pytest.fixture(scope="module")
def model():
    torch.manual_seed(0)
    return PatchTransformerV0().eval()


@pytest.fixture(scope="module")
def source():
    return SyntheticEpisodeSource(load_config(CONFIGS / "train_v0.yaml"), split="debug")


def _batch(episodes):
    return to_torch(collate_episodes(episodes, pad_to_multiple=32))


@torch.no_grad()
def _predict(model, batch, **kw):
    inputs, _ = split_batch(batch)
    return model(**inputs, **kw)["prediction"]


def test_train_v0_config_is_data_v1_without_multivariate():
    v0, v1 = load_config(CONFIGS / "train_v0.yaml"), load_config(CONFIGS / "data_v1.yaml")
    assert v0["multivariate"]["enabled"] is False
    assert v0["univariate_generators"] == v1["univariate_generators"]  # holdout exclusions kept
    assert v0["prior"] == v1["prior"]


def test_parameter_count(model):
    assert 4.5e6 < model.num_parameters() < 5.5e6


def test_output_shape_and_target_channels():
    src = SyntheticEpisodeSource(load_config(CONFIGS / "data_v1.yaml"), split="debug")
    batch = _batch([src.episode(i, family="multivariate") for i in range(6)])
    inputs, _ = split_batch(batch)
    with torch.no_grad():
        out = PatchTransformerV0()(**inputs)
    B, C, W = batch["input_values"].shape
    assert out["prediction"].shape == (B, C, batch["target_future"].shape[-1])
    assert torch.equal(out["target_channels"], batch["channel_roles"] == Role.TARGET)
    assert (out["prediction"][~out["target_channels"]] == 0).all()


def test_labels_are_rejected(model, source):
    batch = _batch([source.episode(0)])
    with pytest.raises(ValueError, match="labels"):
        model(**batch)


def test_future_does_not_change_prediction(model, source):
    batch = _batch([source.episode(i) for i in range(8)])
    ref = _predict(model, batch)
    W = batch["input_values"].shape[-1]
    origin = W - batch["target_future"].shape[-1]
    noisy = dict(batch)
    noisy["input_values"] = batch["input_values"].clone()
    noisy["input_values"][..., origin:] = torch.randn_like(noisy["input_values"][..., origin:]) * 1e3
    noisy["input_mask"] = batch["input_mask"].clone()
    noisy["input_mask"][..., origin:] = True
    noisy["target_future"] = torch.randn_like(batch["target_future"]) * 1e3
    assert torch.equal(_predict(model, noisy), ref)


def test_missing_values_are_ignored(model, source):
    batch = _batch([source.episode(i) for i in range(16)])
    hidden = ~batch["input_mask"]
    assert hidden.any()
    noisy = dict(batch)
    noisy["input_values"] = torch.where(hidden, torch.full_like(batch["input_values"], 1e4), batch["input_values"])
    torch.testing.assert_close(_predict(model, noisy), _predict(model, batch), atol=ATOL, rtol=0)


def test_padding_does_not_change_prediction(model, source):
    eps = [source.episode(i) for i in range(200)]
    short = next(e for e in eps if e.context_length == 128 and e.prediction_length <= 48)
    long = next(e for e in eps if e.context_length == 1024 and e.prediction_length == 192)
    alone = _predict(model, _batch([short]))
    padded = _predict(model, _batch([short, long]))[:1, :, : alone.shape[-1]]
    torch.testing.assert_close(padded, alone, atol=ATOL, rtol=0)


def test_rollout_extends_single_pass(model, source):
    batch = _batch([source.episode(i) for i in range(4)])
    one = _predict(model, batch, horizon=192)
    long = _predict(model, batch, horizon=384)
    assert long.shape[-1] == 384
    torch.testing.assert_close(long[..., :192], one, atol=ATOL, rtol=0)


def test_masked_mae_and_denormalize():
    pred = torch.tensor([[[1.0, 2.0, 9.0]]])
    y = torch.tensor([[[0.0, 0.0]]])
    mask = torch.tensor([[[True, False]]])
    assert masked_mae(pred, y, mask).item() == pytest.approx(1.0)
    stats = torch.tensor([[[10.0, 2.0]]])
    assert torch.equal(denormalize(pred, stats), torch.tensor([[[12.0, 14.0, 28.0]]]))


def test_overfits_a_fixed_batch():
    cfg = load_config(CONFIGS / "train_v0.yaml")
    cfg["sequence"] = {**cfg["sequence"], "context_lengths": {128: 1.0}, "horizons": {48: 1.0}}
    src = SyntheticEpisodeSource(cfg, split="debug")
    batch = _batch([src.episode(i) for i in range(16)])
    inputs, labels = split_batch(batch)
    torch.manual_seed(0)
    model = PatchTransformerV0(PatchV0Config(n_layers=2))
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    losses = []
    for _ in range(300):
        loss = masked_mae(model(**inputs)["prediction"], labels["target_future"], labels["loss_mask"])
        opt.zero_grad()
        loss.backward()
        opt.step()
        losses.append(loss.item())
    assert np.isfinite(losses).all()
    assert losses[-1] < 0.1 * losses[0], (losses[0], losses[-1])
