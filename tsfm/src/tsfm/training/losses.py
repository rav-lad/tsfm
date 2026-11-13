"""Losses in the normalized space, restricted to supervised positions."""

from __future__ import annotations

from typing import Sequence

import torch
from torch import Tensor


def masked_mae(prediction: Tensor, target_future: Tensor, loss_mask: Tensor) -> Tensor:
    """Mean |pred - y| over ``loss_mask``. ``prediction`` may be longer than the
    labels (padded horizon); it is cut to ``target_future``'s length."""
    H = target_future.shape[-1]
    err = (prediction[..., :H] - target_future).abs()
    mask = loss_mask.to(err.dtype)
    return (err * mask).sum() / mask.sum().clamp_min(1.0)


def masked_pinball(quantiles: Tensor, target_future: Tensor, loss_mask: Tensor, levels: Sequence[float],
                   weights: Tensor | None = None) -> Tensor:
    """Pinball loss averaged over quantile levels, then over ``loss_mask``.

    ``quantiles`` [B, C, H', Q] with H' >= H. At level 0.5 the pinball loss is
    0.5 * |e|, so a median-only model gets exactly half of ``masked_mae``.
    ``weights`` (broadcastable to [B, C, H]) gives a weighted mean
    ``sum(w * l) / sum(w)`` over the supervised positions.
    """
    H = target_future.shape[-1]
    q = torch.as_tensor(levels, dtype=quantiles.dtype, device=quantiles.device)
    diff = target_future.unsqueeze(-1) - quantiles[..., :H, :]
    loss = torch.maximum(q * diff, (q - 1) * diff).mean(-1)
    mask = loss_mask.to(loss.dtype)
    if weights is not None:
        mask = mask * weights
    return (loss * mask).sum() / mask.sum().clamp_min(1e-8)


def harmonic_horizon_weights(H: int, device: torch.device | str | None = None) -> Tensor:
    """Timer-M1 horizon weights ``w_h = 1 / h`` (h = 1..H): near steps count more."""
    return 1.0 / torch.arange(1, H + 1, dtype=torch.float32, device=device)


def naive_scale_weights(input_values: Tensor, input_mask: Tensor, padding_mask: Tensor, origin: int,
                        min_scale: float = 0.05) -> Tensor:
    """Per-series weight ``1 / max(mean |x_t - x_{t-1}|, min_scale)`` over the observed
    history (normalized space), so the loss behaves like a MASE: errors count
    relative to the series' one-step variability, not to its level spread.
    Returns [B, C, 1]."""
    x = input_values[..., :origin]
    ok = input_mask[..., :origin] & padding_mask[..., :origin]
    both = ok[..., 1:] & ok[..., :-1]
    d = (x[..., 1:] - x[..., :-1]).abs() * both
    scale = d.sum(-1) / both.sum(-1).clamp_min(1)
    return (1.0 / scale.clamp_min(min_scale)).unsqueeze(-1)


def denormalize(prediction: Tensor, scaling_stats: Tensor) -> Tensor:
    """Back to the raw scale: ``pred * scale + loc``. ``scaling_stats`` [B, C, 2];
    ``prediction`` [B, C, H] or [B, C, H, Q]."""
    loc, scale = scaling_stats[..., 0], scaling_stats[..., 1]
    extra = prediction.ndim - loc.ndim
    loc, scale = loc[(...,) + (None,) * extra], scale[(...,) + (None,) * extra]
    return prediction * scale + loc


__all__ = ["denormalize", "harmonic_horizon_weights", "masked_mae", "masked_pinball", "naive_scale_weights"]
