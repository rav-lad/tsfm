"""Batch collation (numpy). Convert with ``torch.from_numpy`` in the trainer.

Episodes with different (L, H) are aligned on the forecast origin: the context is
left-padded to ``L_max`` and the horizon right-padded to ``H_max``. Padding is
neither observed nor supervised (``padding_mask`` = True on real positions).

``target_future`` is a label: the trainer must pass it to the loss only, never
to ``model.forward``.
"""

from __future__ import annotations

import numpy as np

from .episode import Episode
from .normalizers import RobustScaler


def _round_up(x: int, m: int) -> int:
    return -(-x // m) * m if m > 1 else x


def collate_episodes(
    episodes: list[Episode],
    scaler: RobustScaler | None = None,
    pad_to_multiple: int = 1,
) -> dict[str, np.ndarray]:
    scaler = scaler or RobustScaler()
    B = len(episodes)
    C = max(e.n_channels for e in episodes)
    L_max = _round_up(max(e.context_length for e in episodes), pad_to_multiple)
    H_max = _round_up(max(e.prediction_length for e in episodes), pad_to_multiple)
    W = L_max + H_max

    out = {
        "input_values": np.zeros((B, C, W), np.float32),
        "input_mask": np.zeros((B, C, W), bool),
        "padding_mask": np.zeros((B, C, W), bool),
        "target_future": np.zeros((B, C, H_max), np.float32),
        "loss_mask": np.zeros((B, C, H_max), bool),
        "channel_roles": np.full((B, C), -1, np.int8),  # -1 = padded channel
        "scaling_stats": np.zeros((B, C, 2), np.float32),
        "lengths": np.zeros((B, 2), np.int32),  # (L, H)
    }
    out["scaling_stats"][..., 1] = 1.0
    for b, ep in enumerate(episodes):
        t = scaler.transform(ep)
        L, H, c = ep.context_length, ep.prediction_length, ep.n_channels
        s = L_max - L  # left padding
        out["input_values"][b, :c, s : L_max + H] = t["input_values"]
        out["input_mask"][b, :c, s : L_max + H] = t["input_mask"]
        out["padding_mask"][b, :c, s : L_max + H] = True
        out["target_future"][b, :c, :H] = t["target_future"]
        out["loss_mask"][b, :c, :H] = t["loss_mask"]
        out["channel_roles"][b, :c] = ep.roles
        out["scaling_stats"][b, :c] = t["scaling_stats"]
        out["lengths"][b] = (L, H)
    return out
