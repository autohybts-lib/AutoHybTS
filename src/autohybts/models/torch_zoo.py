"""Example PyTorch architectures for ``TorchForecaster``/``TorchCorrector`` (extra ``torch``).

Every class takes ``n_inputs`` (number of lags) plus its own sizes, and maps a tensor
``(batch, n_inputs)`` of lag values, oldest first, to ``(batch,)``. Importing this module
imports PyTorch; ``import autohybts`` never does.

* ``MLPNet``: dense layers on the lag vector.
* ``LSTMNet``: LSTM over the lags as a sequence; the last hidden state feeds a linear head.
* ``TransformerNet``: Transformer encoder with learned positions and a causal mask.
* ``MambaNet``: a compact Mamba-style selective state-space block (input-dependent
  ``Δ``, ``B``, ``C``; causal depthwise convolution; gated output), written in plain
  PyTorch with a sequential scan. It follows Gu and Dao (2023) but is not the optimized
  ``mamba-ssm`` package, which needs CUDA on Linux; for the short windows of lag models
  the plain scan is enough.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import cast

import torch
from torch import nn

__all__ = ["LSTMNet", "MLPNet", "MambaNet", "TransformerNet"]


class MLPNet(nn.Module):
    def __init__(
        self, n_inputs: int, hidden: Sequence[int] = (64, 32), dropout: float = 0.0
    ) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        width = n_inputs
        for size in hidden:
            layers += [nn.Linear(width, int(size)), nn.ReLU(), nn.Dropout(dropout)]
            width = int(size)
        layers.append(nn.Linear(width, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return cast(torch.Tensor, self.net(x).squeeze(-1))


class LSTMNet(nn.Module):
    def __init__(
        self, n_inputs: int, hidden: int = 32, layers: int = 1, dropout: float = 0.0
    ) -> None:
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=1,
            hidden_size=hidden,
            num_layers=layers,
            batch_first=True,
            dropout=dropout if layers > 1 else 0.0,
        )
        self.head = nn.Linear(hidden, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.lstm(x.unsqueeze(-1))
        return cast(torch.Tensor, self.head(out[:, -1]).squeeze(-1))


class TransformerNet(nn.Module):
    def __init__(
        self,
        n_inputs: int,
        d_model: int = 32,
        heads: int = 4,
        layers: int = 2,
        feedforward: int = 64,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.embed = nn.Linear(1, d_model)
        self.position = nn.Parameter(torch.zeros(1, n_inputs, d_model))
        layer = nn.TransformerEncoderLayer(
            d_model, heads, feedforward, dropout=dropout, batch_first=True
        )
        self.encoder = nn.TransformerEncoder(layer, layers)
        self.head = nn.Linear(d_model, 1)
        mask = torch.triu(torch.full((n_inputs, n_inputs), float("-inf")), diagonal=1)
        self.register_buffer("mask", mask)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.embed(x.unsqueeze(-1)) + self.position
        h = self.encoder(h, mask=self.mask)
        return cast(torch.Tensor, self.head(h[:, -1]).squeeze(-1))


class _SelectiveSSM(nn.Module):
    """One Mamba-style block: in-projection, causal conv, selective scan, gate, out-projection."""

    def __init__(self, d_model: int, d_state: int, expand: int, conv: int) -> None:
        super().__init__()
        inner = expand * d_model
        self.inner, self.d_state = inner, d_state
        self.in_proj = nn.Linear(d_model, 2 * inner)
        self.conv = nn.Conv1d(inner, inner, conv, groups=inner, padding=conv - 1)
        self.x_proj = nn.Linear(inner, 1 + 2 * d_state)  # Δ (scalar per step), B, C
        self.dt_proj = nn.Linear(1, inner)
        a = torch.arange(1, d_state + 1, dtype=torch.float32).repeat(inner, 1)
        self.a_log = nn.Parameter(torch.log(a))
        self.d = nn.Parameter(torch.ones(inner))
        self.out_proj = nn.Linear(inner, d_model)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        length = h.shape[1]
        x, z = self.in_proj(self.norm(h)).chunk(2, dim=-1)
        x = self.conv(x.transpose(1, 2))[..., :length].transpose(1, 2)  # causal
        x = nn.functional.silu(x)
        dt, b, c = torch.split(self.x_proj(x), [1, self.d_state, self.d_state], dim=-1)
        delta = nn.functional.softplus(self.dt_proj(dt))  # (batch, length, inner)
        a = -torch.exp(self.a_log)  # (inner, d_state)
        # Selective scan, all steps at once: state_t = Σ_{s<=t} exp(L_t - L_s) · u_s with
        # L the cumulative log-decay. Same result as the step-by-step recursion
        # state_t = decay_t · state_{t-1} + u_t, but without a Python loop (one kernel
        # launch per op instead of one per step, which matters on GPU). Every exponent is
        # <= 0, so it is stable; the causal mask keeps step t from seeing s > t.
        log_decay = delta[..., None] * a  # (batch, length, inner, d_state), <= 0
        cumulative = torch.cumsum(log_decay, dim=1)
        gaps = cumulative[:, :, None] - cumulative[:, None, :]  # (batch, t, s, inner, d_state)
        causal = torch.ones(length, length, dtype=torch.bool, device=h.device).tril()
        gaps = gaps.masked_fill(~causal[None, :, :, None, None], float("-inf"))
        inputs = delta[..., None] * b[:, :, None, :] * x[..., None]  # u_s
        state = torch.einsum("btsij,bsij->btij", torch.exp(gaps), inputs)
        y = (state * c[:, :, None, :]).sum(-1) + self.d * x
        y = y * nn.functional.silu(z)
        return cast(torch.Tensor, h + self.out_proj(y))


class MambaNet(nn.Module):
    def __init__(
        self,
        n_inputs: int,
        d_model: int = 32,
        d_state: int = 16,
        layers: int = 2,
        expand: int = 2,
        conv: int = 4,
    ) -> None:
        super().__init__()
        self.embed = nn.Linear(1, d_model)
        self.blocks = nn.ModuleList(
            _SelectiveSSM(d_model, d_state, expand, conv) for _ in range(layers)
        )
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.embed(x.unsqueeze(-1))
        for block in self.blocks:
            h = block(h)
        return cast(torch.Tensor, self.head(self.norm(h[:, -1])).squeeze(-1))
