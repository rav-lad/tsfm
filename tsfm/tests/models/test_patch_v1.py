"""Model V1 (quantile head): outputs, guard rails, pinball semantics, overfit, checkpoint."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import torch

from tsfm.data import SyntheticEpisodeSource, collate_episodes, load_config
from tsfm.models import MODEL_PRESETS, V1_QUANTILES, PatchTransformerV0, PatchV0Config
from tsfm.training import split_batch, to_torch
from tsfm.training.losses import denormalize, masked_mae, masked_pinball

CONFIGS = Path(__file__).parents[2] / "configs"
ATOL = 1e-5
Q = len(V1_QUANTILES)


@pytest.fixture(scope="module")
def model():
    torch.manual_seed(0)
    return PatchTransformerV0(MODEL_PRESETS["v1"]).eval()


@pytest.fixture(scope="module")
def source():
    return SyntheticEpisodeSource(load_config(CONFIGS / "train_v0.yaml"), split="debug")


def _batch(episodes):
    return to_torch(collate_episodes(episodes, pad_to_multiple=32))


@torch.no_grad()
def _run(model, batch, **kw):
    inputs, _ = split_batch(batch)
    return model(**inputs, **kw)


def test_parameter_count(model):
    assert 6.0e6 < model.num_parameters() < 7.0e6


def test_outputs_median_and_sorted_quantiles(model, source):
    batch = _batch([source.episode(i) for i in range(8)])
    out = _run(model, batch)
    B, C, _ = batch["input_values"].shape
    H = batch["target_future"].shape[-1]
    assert out["quantiles"].shape == (B, C, H, Q)
    assert torch.equal(out["prediction"], out["quantiles"][..., V1_QUANTILES.index(0.5)])
    assert (out["quantiles"].diff(dim=-1) >= 0).all()  # no crossing at inference


def test_v0_has_no_quantiles(source):
    out = _run(PatchTransformerV0(MODEL_PRESETS["v0"]).eval(), _batch([source.episode(0)]))
    assert "quantiles" not in out


@pytest.mark.parametrize("bad", [(0.1, 0.9), (0.5, 0.1), (0.0, 0.5), (0.5, 0.5)])
def test_invalid_quantiles_rejected(bad):
    with pytest.raises(ValueError, match="quantiles"):
        PatchTransformerV0(PatchV0Config(quantiles=bad))


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


def test_padding_does_not_change_quantiles(model, source):
    eps = [source.episode(i) for i in range(200)]
    short = next(e for e in eps if e.context_length == 128 and e.prediction_length <= 48)
    long = next(e for e in eps if e.context_length == 1024 and e.prediction_length == 192)
    alone = _run(model, _batch([short]))["quantiles"]
    padded = _run(model, _batch([short, long]))["quantiles"][:1, :, : alone.shape[2]]
    torch.testing.assert_close(padded, alone, atol=ATOL, rtol=0)


def test_rollout_feeds_the_median(model, source):
    batch = _batch([source.episode(i) for i in range(4)])
    one = _run(model, batch, horizon=192)
    long = _run(model, batch, horizon=400)
    assert long["quantiles"].shape[2] == 400
    torch.testing.assert_close(long["quantiles"][:, :, :192], one["quantiles"], atol=ATOL, rtol=0)


def test_pinball_at_median_is_half_mae():
    torch.manual_seed(0)
    pred, y = torch.randn(2, 3, 10), torch.randn(2, 3, 8)
    mask = torch.rand(2, 3, 8) > 0.3
    assert masked_pinball(pred[..., None], y, mask, (0.5,)).item() == pytest.approx(
        0.5 * masked_mae(pred, y, mask).item(), rel=1e-6)


def test_pinball_minimizer_is_the_quantile():
    """A free constant per level, fit with the pinball loss on N(0, 1) samples,
    converges to the Gaussian quantiles: the loss teaches the right thing."""
    torch.manual_seed(0)
    y = torch.randn(1, 1, 20_000)
    mask = torch.ones_like(y, dtype=torch.bool)
    q = torch.zeros(1, 1, 1, Q, requires_grad=True)
    opt = torch.optim.Adam([q], lr=0.05)
    for _ in range(600):
        loss = masked_pinball(q.expand(1, 1, y.shape[-1], Q), y, mask, V1_QUANTILES)
        opt.zero_grad()
        loss.backward()
        opt.step()
    expected = torch.distributions.Normal(0, 1).icdf(torch.tensor(V1_QUANTILES))
    torch.testing.assert_close(q.detach().flatten(), expected, atol=0.05, rtol=0)


def test_denormalize_quantiles():
    stats = torch.tensor([[[10.0, 2.0]]])
    q = torch.ones(1, 1, 3, 2)
    assert torch.equal(denormalize(q, stats), torch.full((1, 1, 3, 2), 12.0))


def test_overfits_a_fixed_batch_with_pinball():
    cfg = load_config(CONFIGS / "train_v0.yaml")
    cfg["sequence"] = {**cfg["sequence"], "context_lengths": {128: 1.0}, "horizons": {48: 1.0}}
    src = SyntheticEpisodeSource(cfg, split="debug")
    inputs, labels = split_batch(_batch([src.episode(i) for i in range(16)]))
    torch.manual_seed(0)
    model = PatchTransformerV0(replace(MODEL_PRESETS["v1"], n_layers=2))
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    losses = []
    for _ in range(300):
        out = model(**inputs)
        loss = masked_pinball(out["quantiles"], labels["target_future"], labels["loss_mask"], V1_QUANTILES)
        opt.zero_grad()
        loss.backward()
        opt.step()
        losses.append(loss.item())
    assert np.isfinite(losses).all()
    assert losses[-1] < 0.1 * losses[0], (losses[0], losses[-1])
