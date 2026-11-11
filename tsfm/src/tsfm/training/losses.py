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


def masked_pinball(quantiles: Tensor, target_future: Tensor, loss_mask: Tensor, levels: Sequence[float]) -> Tensor:
    """Pinball loss averaged over quantile levels, then over ``loss_mask``.

    ``quantiles`` [B, C, H', Q] with H' >= H. At level 0.5 the pinball loss is
    0.5 * |e|, so a median-only model gets exactly half of ``masked_mae``.
    """
    H = target_future.shape[-1]
    q = torch.as_tensor(levels, dtype=quantiles.dtype, device=quantiles.device)
    diff = target_future.unsqueeze(-1) - quantiles[..., :H, :]
    loss = torch.maximum(q * diff, (q - 1) * diff).mean(-1)
    mask = loss_mask.to(loss.dtype)
    return (loss * mask).sum() / mask.sum().clamp_min(1.0)


def denormalize(prediction: Tensor, scaling_stats: Tensor) -> Tensor:
    """Back to the raw scale: ``pred * scale + loc``. ``scaling_stats`` [B, C, 2];
    ``prediction`` [B, C, H] or [B, C, H, Q]."""
    loc, scale = scaling_stats[..., 0], scaling_stats[..., 1]
    extra = prediction.ndim - loc.ndim
    loc, scale = loc[(...,) + (None,) * extra], scale[(...,) + (None,) * extra]
    return prediction * scale + loc


__all__ = ["denormalize", "masked_mae", "masked_pinball"]
