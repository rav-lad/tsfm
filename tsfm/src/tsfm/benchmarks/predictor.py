"""Wrap a trained model as a benchmark predictor (raw contexts -> raw forecasts).

Each context goes through the training pipeline: last ``max_context`` points,
``Episode`` with NaN = not observed, ``collate_episodes`` (median/MAD on the
visible history, left padding to whole patches), model forward with
``horizon = prediction_length`` (rollout beyond ``max_horizon``), denormalize.
A quantile model (V1) returns ``(median, quantiles)``; its levels must be the
GIFT-Eval ones (``V1_QUANTILES``), so no interpolation is involved.
"""

from __future__ import annotations

import numpy as np
import torch
from torch import nn

from tsfm.data import Episode, Role, collate_episodes
from tsfm.training import split_batch, to_torch
from tsfm.training.losses import denormalize

from .gift_eval import QUANTILES, Predictor


def context_episode(context: np.ndarray, prediction_length: int, max_context: int) -> Episode:
    hist = np.asarray(context, np.float32)[-max_context:]
    L, H = len(hist), prediction_length
    observed = np.zeros((1, L + H), bool)
    observed[0, :L] = np.isfinite(hist)
    values = np.zeros((1, L + H), np.float32)
    values[0, :L] = np.where(observed[0, :L], hist, 0.0)
    return Episode(values=values, observed_mask=observed, loss_mask=np.zeros((1, H), bool),
                   roles=np.array([Role.TARGET], np.int8), context_length=L, prediction_length=H, origin=L)


def model_predictor(model: nn.Module, device: str | torch.device = "cpu", batch_size: int = 256,
                    max_context: int = 1024, pad_to_multiple: int = 32) -> Predictor:
    model = model.to(device).eval()
    levels = tuple(getattr(model, "quantile_levels", ()) or ())
    if levels and (len(levels) != len(QUANTILES) or not np.allclose(levels, QUANTILES)):
        raise ValueError(f"model quantiles {levels} != GIFT-Eval levels {tuple(QUANTILES)}")

    @torch.no_grad()
    def predict(contexts: list[np.ndarray], prediction_length: int, seasonality: int):
        out = np.zeros((len(contexts), prediction_length), np.float32)
        quant = np.zeros((len(contexts), prediction_length, len(levels)), np.float32) if levels else None
        # similar lengths together: less left padding per batch
        order = np.argsort([min(len(c), max_context) for c in contexts], kind="stable")
        for start in range(0, len(order), batch_size):
            idx = order[start : start + batch_size]
            eps = [context_episode(contexts[i], prediction_length, max_context) for i in idx]
            inputs, _ = split_batch(to_torch(collate_episodes(eps, pad_to_multiple=pad_to_multiple), device))
            res = model(**inputs, horizon=prediction_length)
            out[idx] = denormalize(res["prediction"].float(), inputs["scaling_stats"])[:, 0].cpu().numpy()
            if quant is not None:
                quant[idx] = denormalize(res["quantiles"].float(), inputs["scaling_stats"])[:, 0].cpu().numpy()
        return out if quant is None else (out, quant)

    return predict
