from typing import Optional

import torch
import torch.nn as nn


class Time2Vec(nn.Module):
    def __init__(self, out_dim: int):
        super().__init__()

        if out_dim < 2:
            raise ValueError("out_dim must be at least 2")

        periodic_dim = out_dim - 1

        self.w0 = nn.Parameter(torch.randn(1) * 0.01)
        self.b0 = nn.Parameter(torch.zeros(1))
        self.W = nn.Parameter(torch.randn(periodic_dim) * 0.01)
        self.B = nn.Parameter(torch.zeros(periodic_dim))

    def forward(self, t: torch.Tensor) -> torch.Tensor:

        t = t.float().unsqueeze(-1)
        linear = t * self.w0 + self.b0
        periodic = torch.sin(t * self.W + self.B)

        return torch.cat([linear, periodic],dim=-1)


class TimeEncoder(nn.Module):
    def __init__(self, hidden_dim: int):
        super().__init__()

        half_dim = hidden_dim // 2

        self.absolute = Time2Vec(half_dim)
        self.relative = Time2Vec(half_dim)

        self.projection = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim))

    def forward(
        self,
        t_abs: torch.Tensor,
        t_rel: Optional[torch.Tensor] = None) -> torch.Tensor:

        absolute = self.absolute(t_abs)

        if t_rel is None:
            relative = torch.zeros_like(absolute)
        else:
            relative = self.relative(t_rel)

        encoded_time = torch.cat([absolute, relative],dim=-1)
        return self.projection(encoded_time)


class TRME(nn.Module):
    """Applies continuous-time TRME to messages."""

    def __init__(self, dim: int):
        super().__init__()

        if dim % 2 != 0:
            raise ValueError("dim must be even for RoPE")

        self.log_timescales = nn.Parameter(
            torch.linspace(0, 6, dim // 2))

    def forward(
        self,
        message: torch.Tensor,
        timestamp: torch.Tensor) -> torch.Tensor:

        timescales = self.log_timescales.exp()
        angles = (timestamp.float().unsqueeze(-1)/ timescales)

        cos = angles.cos()
        sin = angles.sin()
        even = message[:, 0::2]
        odd = message[:, 1::2]
        rotated_even = even * cos - odd * sin
        rotated_odd = even * sin + odd * cos

        return torch.stack(
                    [rotated_even, rotated_odd], dim=-1,
                          ).reshape_as(message)