"""Timestamp encoders for the T-ADTG encoder."""

import torch
import torch.nn as nn


class Time2Vec(nn.Module):
    """Learnable time encoding: one linear (trend) term plus k sinusoidal terms."""

    def __init__(self, out_dim: int):
        super().__init__()
        assert out_dim >= 2
        k = out_dim - 1
        self.w0 = nn.Parameter(torch.randn(1) * 0.01)
        self.b0 = nn.Parameter(torch.zeros(1))
        self.W = nn.Parameter(torch.randn(k) * 0.01)
        self.B = nn.Parameter(torch.zeros(k))

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        t = t.float().unsqueeze(-1)
        lin = t * self.w0 + self.b0
        per = torch.sin(t * self.W + self.B)
        return torch.cat([lin, per], dim=-1)


class TimeEncoder(nn.Module):
    """Encodes absolute and relative timestamps into a single hidden vector.

    Absolute time captures global position in the stream, relative time the
    age of an edge with respect to the root's latest event.
    """

    def __init__(self, hidden_dim: int):
        super().__init__()
        half = hidden_dim // 2
        self.t2v_abs = Time2Vec(half)
        self.t2v_rel = Time2Vec(hidden_dim - half)
        self.proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(self, t_abs: torch.Tensor, t_rel: torch.Tensor) -> torch.Tensor:
        a = self.t2v_abs(t_abs)
        r = self.t2v_rel(t_rel)
        return self.proj(torch.cat([a, r], dim=-1))