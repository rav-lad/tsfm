"""Models V0 / V1 / V2: univariate patch Transformer, direct multi-horizon forecast.

- V0: point forecast (``quantiles=()``), trained with MAE.
- V1: quantile forecast (``quantiles=V1_QUANTILES``), trained with the pinball
  loss; ``prediction`` is the median, ``quantiles`` the full set.
- V2: V1 with future patch tokens (``future_tokens=True``, Chronos-2 / Timestral
  DHF-like) and ``max_horizon=1024``: one learned token per future patch is
  appended after the history, every token attends to every other one, and each
  future token is decoded into its own ``patch_size x Q`` block. The horizon is
  produced in one pass up to ``max_horizon`` (no rollout), and the number of
  future tokens follows the requested horizon.

Roadmap phase 2 / V0-V1 (Timestral-5M-like sizes)::

    TARGET channels, history only          [N, L]
    -> patches of 32 (values ++ mask)      [N, P, 64]
    -> residual MLP embedding              [N, P, d]
    -> n_layers x (RMSNorm, attention + RoPE, RMSNorm, SwiGLU), bidirectional
    -> last history patch -> RMSNorm -> residual MLP head -> max_horizon x Q
       (V2: ++ future tokens before the blocks; each -> head -> patch_size x Q)

Everything stays in the normalized space produced by ``collate_episodes``.
The model only reads ``input_values[..., :origin]``: the future part of the
batch is never touched, so target leakage is impossible by construction.
Horizons longer than ``max_horizon`` are produced by rollout; a quantile model
feeds its median back as history, so quantiles past the first ``max_horizon``
steps are conditional on that median path (uncertainty is underestimated).
At inference (``model.eval()``) quantiles are sorted to remove crossings.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from tsfm.data import Role

FORBIDDEN_INPUTS = ("target_future", "loss_mask")
# the 9 levels of GIFT-Eval's CRPS (mean weighted quantile loss): no interpolation needed
V1_QUANTILES = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)


@dataclass
class PatchV0Config:
    patch_size: int = 32
    d_model: int = 192
    n_layers: int = 8
    n_heads: int = 3
    ffn_hidden: int = 768
    embed_hidden: int = 384
    head_hidden: int = 768
    max_horizon: int = 192
    rope_base: float = 10_000.0
    quantiles: tuple[float, ...] = ()  # () = point forecast (V0)
    future_tokens: bool = False  # V2: one token per future patch instead of a last-token head


class ResidualMLP(nn.Module):
    def __init__(self, d_in: int, d_hidden: int, d_out: int) -> None:
        super().__init__()
        self.fc1 = nn.Linear(d_in, d_hidden)
        self.fc2 = nn.Linear(d_hidden, d_out)
        self.skip = nn.Linear(d_in, d_out)

    def forward(self, x: Tensor) -> Tensor:
        return self.fc2(F.silu(self.fc1(x))) + self.skip(x)


class SwiGLU(nn.Module):
    def __init__(self, d: int, hidden: int) -> None:
        super().__init__()
        self.w1 = nn.Linear(d, hidden, bias=False)
        self.w3 = nn.Linear(d, hidden, bias=False)
        self.w2 = nn.Linear(hidden, d, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


def rope(x: Tensor, positions: Tensor, base: float) -> Tensor:
    """Rotary embedding on the last dim of ``x`` [N, heads, P, head_dim]."""
    half = x.shape[-1] // 2
    freqs = base ** (-torch.arange(half, device=x.device, dtype=torch.float32) / half)
    angles = positions.to(torch.float32)[:, None] * freqs  # [P, half]
    cos, sin = angles.cos().to(x.dtype), angles.sin().to(x.dtype)
    x1, x2 = x[..., :half], x[..., half:]
    return torch.cat([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1)


class Attention(nn.Module):
    def __init__(self, cfg: PatchV0Config) -> None:
        super().__init__()
        if cfg.d_model % cfg.n_heads:
            raise ValueError("d_model must be divisible by n_heads")
        self.n_heads = cfg.n_heads
        self.rope_base = cfg.rope_base
        self.qkv = nn.Linear(cfg.d_model, 3 * cfg.d_model, bias=False)
        self.out = nn.Linear(cfg.d_model, cfg.d_model, bias=False)

    def forward(self, h: Tensor, positions: Tensor, key_mask: Tensor) -> Tensor:
        N, P, D = h.shape
        q, k, v = self.qkv(h).view(N, P, 3, self.n_heads, D // self.n_heads).permute(2, 0, 3, 1, 4)
        q, k = rope(q, positions, self.rope_base), rope(k, positions, self.rope_base)
        # key_mask [N, P]: True = real patch. Bidirectional: the whole history is known.
        o = F.scaled_dot_product_attention(q, k, v, attn_mask=key_mask[:, None, None, :])
        return self.out(o.transpose(1, 2).reshape(N, P, D))


class Block(nn.Module):
    def __init__(self, cfg: PatchV0Config) -> None:
        super().__init__()
        self.norm1 = nn.RMSNorm(cfg.d_model)
        self.attn = Attention(cfg)
        self.norm2 = nn.RMSNorm(cfg.d_model)
        self.ffn = SwiGLU(cfg.d_model, cfg.ffn_hidden)

    def forward(self, h: Tensor, positions: Tensor, key_mask: Tensor) -> Tensor:
        h = h + self.attn(self.norm1(h), positions, key_mask)
        return h + self.ffn(self.norm2(h))


def forecast_origin(padding_mask: Tensor, lengths: Tensor) -> int:
    """Index of the first future step in the collated window (same for all rows)."""
    first_real = padding_mask[0, 0].to(torch.int8).argmax()
    return int(first_real + lengths[0, 0])


class PatchTransformerV0(nn.Module):
    def __init__(self, cfg: PatchV0Config | None = None) -> None:
        super().__init__()
        self.cfg = cfg = cfg or PatchV0Config()
        if cfg.max_horizon % cfg.patch_size:
            raise ValueError("max_horizon must be a multiple of patch_size (rollout appends whole patches)")
        q = tuple(float(x) for x in cfg.quantiles)
        if q and (list(q) != sorted(set(q)) or 0.5 not in q or not all(0 < x < 1 for x in q)):
            raise ValueError(f"quantiles must be sorted, unique, in (0, 1) and contain 0.5: {q}")
        self.quantile_levels = q
        self.n_outputs = len(q) or 1
        self.median_index = q.index(0.5) if q else 0
        self.embed = ResidualMLP(2 * cfg.patch_size, cfg.embed_hidden, cfg.d_model)
        self.blocks = nn.ModuleList(Block(cfg) for _ in range(cfg.n_layers))
        self.norm = nn.RMSNorm(cfg.d_model)
        if cfg.future_tokens:
            self.future_embed = nn.Parameter(0.02 * torch.randn(cfg.d_model))
            self.head = ResidualMLP(cfg.d_model, cfg.head_hidden, cfg.patch_size * self.n_outputs)
        else:
            self.head = ResidualMLP(cfg.d_model, cfg.head_hidden, cfg.max_horizon * self.n_outputs)

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())

    # ------------------------------------------------------------------ core
    def forecast_series(self, x: Tensor, observed: Tensor, real: Tensor, horizon: int | None = None) -> Tensor:
        """One pass. ``x, observed, real``: [N, T] history ending at the origin
        (``real`` = not padding). Returns [N, max_horizon, Q] (Q = 1 for V0); with
        future tokens, [N, K * patch_size, Q] where K = ceil(horizon / patch_size)
        (at most max_horizon / patch_size)."""
        p = self.cfg.patch_size
        pad = (-x.shape[1]) % p
        if pad:  # left-pad so the last patch ends exactly at the origin
            x, observed, real = (F.pad(t, (pad, 0)) for t in (x, observed, real))
        N, T = x.shape
        P = T // p
        obs = observed & real
        x = torch.where(obs, x, torch.zeros_like(x))
        tokens = torch.cat([x.view(N, P, p), obs.view(N, P, p).to(x.dtype)], dim=-1)
        key_mask = real.view(N, P, p).any(-1)
        # positions counted back from the origin: left padding never shifts real patches
        h = self.embed(tokens)
        K = 0
        if self.cfg.future_tokens:  # future patches sit at positions 1..K after the origin
            K = min(-(-(horizon or self.cfg.max_horizon) // p), self.cfg.max_horizon // p)
            h = torch.cat([h, self.future_embed.to(h.dtype).expand(N, K, -1)], 1)
            key_mask = torch.cat([key_mask, key_mask.new_ones(N, K)], 1)
        positions = torch.arange(P + K, device=x.device) - (P - 1)

        for block in self.blocks:
            h = block(h, positions, key_mask)
        if K:
            out = self.head(self.norm(h[:, P:])).view(N, K * p, self.n_outputs)
        else:
            out = self.head(self.norm(h[:, -1])).view(N, self.cfg.max_horizon, self.n_outputs)
        if self.quantile_levels and not self.training:
            out = out.sort(dim=-1).values  # no quantile crossing at inference
        return out

    def forward(
        self,
        input_values: Tensor,
        input_mask: Tensor,
        padding_mask: Tensor,
        channel_roles: Tensor,
        lengths: Tensor,
        horizon: int | None = None,
        **_unused: Tensor,
    ) -> dict[str, Tensor]:
        """Batch from ``split_batch`` -> ``prediction`` [B, C, horizon] (normalized
        space, zero on non-target channels; the median for a quantile model),
        ``target_channels`` [B, C] and, for a quantile model, ``quantiles``
        [B, C, horizon, Q]."""
        leaked = [k for k in FORBIDDEN_INPUTS if k in _unused]
        if leaked:
            raise ValueError(f"labels passed to model.forward: {leaked} (use split_batch)")
        origin = forecast_origin(padding_mask, lengths)
        if horizon is None:
            horizon = input_values.shape[-1] - origin

        targets = channel_roles == Role.TARGET
        x = input_values[..., :origin][targets]
        observed = input_mask[..., :origin][targets]
        real = padding_mask[..., :origin][targets]

        steps: list[Tensor] = []
        done = 0
        while done < horizon:
            y = self.forecast_series(x, observed, real, horizon - done)
            steps.append(y)
            done += y.shape[1]
            if done < horizon:  # rollout: the point / median becomes observed history
                y_point = y[..., self.median_index]
                ones = torch.ones_like(y_point, dtype=torch.bool)
                x = torch.cat([x, y_point], -1)
                observed, real = torch.cat([observed, ones], -1), torch.cat([real, ones], -1)
        pred = torch.cat(steps, 1)[:, :horizon]  # [N, horizon, Q]

        B, C = channel_roles.shape
        full = pred.new_zeros(B, C, horizon, self.n_outputs)
        full[targets] = pred
        out = {"prediction": full[..., self.median_index], "target_channels": targets}
        if self.quantile_levels:
            out["quantiles"] = full
        return out
