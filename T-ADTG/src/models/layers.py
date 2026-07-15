"""Trust-modulated temporal attention layer."""

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import softmax


class TrustTemporalAttention(MessagePassing):
    """Cross-attention message passing with a per-edge trust gate.

    For each root i and incoming neighbour j, the query comes from the root
    and the key/value from the neighbour context (features + edge features +
    time encoding). Attention weights are normalised per receiver, then
    rescaled by a trust factor in (0, 2) computed from live representations:
    values above 1 amplify agreeing neighbours, values below 1 attenuate
    disagreeing ones. Trust participates in the gradient path and is also
    cached (detached) for the structural loss and the anomaly score.
    """

    def __init__(self, dim: int, heads: int = 8, dropout: float = 0.1):
        super().__init__(aggr="add", node_dim=0, flow="source_to_target")
        assert dim % heads == 0, "dim must be divisible by heads"
        self.dim = dim
        self.heads = heads
        self.head_dim = dim // heads
        self.scale = self.head_dim ** -0.5

        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)

        h = max(dim // 4, 16)
        self.trust_mlp = nn.Sequential(nn.Linear(3, h), nn.GELU(), nn.Linear(h, 1))

        self.gate = nn.Linear(dim, 1)
        self.norm = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(dropout)

        self._trust_cache: Optional[torch.Tensor] = None

    def forward(self, x, edge_index, edge_emb, time_emb, edge_time_feat, deg):
        N = x.size(0)

        # The sampler emits edges as [root, neighbour]. PyG aggregates into the
        # target node, so we flip the rows to route messages neighbour -> root.
        # Column order is untouched, keeping per-edge tensors and the cached
        # trust aligned with the original edge_index.
        rev = edge_index[[1, 0]]

        out = self.propagate(
            rev,
            x=x,
            edge_emb=edge_emb,
            time_emb=time_emb,
            edge_time_feat=edge_time_feat,
            deg=deg,
            size=(N, N),
        )

        # Gated residual: preserve the node's own signal, let structure steer.
        g = torch.sigmoid(self.gate(x))
        return self.norm(g * x + (1.0 - g) * out)

    def message(self, x_i, x_j, edge_emb, time_emb, edge_time_feat,
                deg_i, deg_j, index):
        E = x_i.size(0)

        ctx = x_j + edge_emb + time_emb

        Q = self.q_proj(x_i).view(E, self.heads, self.head_dim)
        K = self.k_proj(ctx).view(E, self.heads, self.head_dim)
        V = self.v_proj(ctx).view(E, self.heads, self.head_dim)

        scores = (Q * K).sum(-1) * self.scale
        scores = softmax(scores, index)  # per-receiver, per-head
        scores = self.dropout(scores)

        # Trust gate from representation agreement, degree similarity and
        # a standardised temporal cue.
        emb_cos = F.cosine_similarity(x_i, x_j, dim=-1).unsqueeze(-1)
        di, dj = deg_i.float(), deg_j.float()
        deg_sim = (1.0 - (di - dj).abs() / (di + dj + 1e-6)).unsqueeze(-1)
        dt = edge_time_feat.unsqueeze(-1)
        trust = 1.0 + torch.tanh(
            self.trust_mlp(torch.cat([emb_cos, deg_sim, dt], dim=-1)).squeeze(-1)
        )
        self._trust_cache = trust.detach()

        msg = (V * scores.unsqueeze(-1)).reshape(E, self.dim)
        return msg * trust.unsqueeze(-1)