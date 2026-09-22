from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing, TransformerConv
from torch_geometric.utils import softmax

from .time_encoder import TRME



class History(nn.Module):
    """Circular buffer containing historical node embeddings."""

    def __init__(self,
                 num_nodes: int,
                 dim: int,
                 window: int = 8):
        super().__init__()
        self.window = window
        self.register_buffer("bank",torch.zeros(num_nodes, window, dim))
        self.register_buffer("times",torch.zeros(num_nodes, window))
        self.register_buffer("ptr", torch.zeros(num_nodes, dtype=torch.long))

    def read(self,idx: torch.Tensor,) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.bank[idx], self.times[idx]

    @torch.no_grad()
    def write(
        self,
        idx: torch.Tensor,
        embeddings: torch.Tensor,
        timestamps: torch.Tensor ):

        for b in range(idx.size(0)):
            node = int(idx[b])
            position = int(self.ptr[node]) % self.window
            self.bank[node, position] = embeddings[b].detach()
            self.times[node, position] = timestamps[b]
            self.ptr[node] += 1

    @torch.no_grad()
    def reset(self):
        self.bank.zero_()
        self.times.zero_()
        self.ptr.zero_()


class GNN(nn.Module):
    """Encodes the current graph snapshot."""

    def __init__(
        self,
        dim: int,
        heads: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()

        if dim % heads != 0:
            raise ValueError("dim must be divisible by heads")

        self.conv1 = TransformerConv(
            in_channels=dim,
            out_channels=dim // heads,
            heads=heads,
            dropout=dropout,
            edge_dim=dim,
            concat=True,
            beta=True,
        )

        self.conv2 = TransformerConv(
            in_channels=dim,
            out_channels=dim // heads,
            heads=heads,
            dropout=dropout,
            edge_dim=dim,
            concat=True,
            beta=True,
        )

        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.output_projection = nn.Linear(dim, dim)
        self.gate = nn.Linear(dim, 1)

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor) -> torch.Tensor:

        h = self.conv1(x,edge_index,edge_attr)
        h = self.norm1(F.gelu(h))
        h2 = self.conv2(h,edge_index,edge_attr)
        h2 = self.norm2(F.gelu(h2))
        h = h + h2
        h = self.output_projection(h)
        gate = torch.sigmoid(self.gate(x))

        return gate * x + (1.0 - gate) * h


class Similarity(nn.Module):
    """Computes similarity from historical trajectory."""

    def forward(
        self,
        embeddings_i: torch.Tensor,
        embeddings_j: torch.Tensor) -> torch.Tensor:
        mean_i = embeddings_i.mean(dim=1)
        mean_j = embeddings_j.mean(dim=1)
        similarity = F.cosine_similarity(mean_i,mean_j,dim=-1)
        return 1.0 + torch.tanh(similarity)




class SATGAT(MessagePassing):

    """
    Temporal trust-aware attention layer.

    """

    def __init__(self, dim: int, heads: int = 8, dropout: float = 0.1):

        super().__init__(aggr="add",node_dim=0,)
        if dim % heads != 0:
            raise ValueError("dim must be divisible by heads")

        self.dim = dim
        self.heads = heads
        self.head_dim = dim // heads
        self.scale = self.head_dim ** -0.5

        self.q_proj = nn.Linear(dim,dim)
        self.k_proj = nn.Linear(3 * dim,dim)
        self.v_proj = nn.Linear(3 * dim,dim)
        self.output_projection = nn.Linear(dim,dim)

        self.message_rope = TRME(dim)
        self.trust = Similarity()
        self.dropout = nn.Dropout(dropout)
        self.gate = nn.Linear(dim, 1)
        self.norm = nn.LayerNorm(dim)

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_emb: torch.Tensor,
        time_emb: torch.Tensor,
        edge_t: torch.Tensor,
        history: History) -> torch.Tensor:

        num_nodes = x.size(0)
        node_idx = torch.arange(num_nodes,device=x.device)
        historical_embeddings, _ = history.read(node_idx)

        out = self.propagate(
            edge_index,
            x=x,
            edge_emb=edge_emb,
            time_emb=time_emb,
            historical_embeddings=historical_embeddings,
            edge_t=edge_t,
            size=(num_nodes, num_nodes))

        out = out.view(num_nodes,self.dim)
        gate = torch.sigmoid(self.gate(x))
        return self.norm(gate * x+ (1.0 - gate) * out)

    def message(
        self,
        x_i: torch.Tensor,
        x_j: torch.Tensor,
        edge_emb: torch.Tensor,
        time_emb: torch.Tensor,
        historical_embeddings_i: torch.Tensor,
        historical_embeddings_j: torch.Tensor,
        edge_t: torch.Tensor,
        index: torch.Tensor) -> torch.Tensor:

        num_edges = x_i.size(0)

        query = self.q_proj(x_j)
        query = query.view(num_edges,self.heads,self.head_dim)

        kv_input = torch.cat([x_i,edge_emb,time_emb,],dim=-1)
        key = self.k_proj(kv_input)
        value = self.v_proj(kv_input)

        key = key.view( num_edges, self.heads, self.head_dim)
        value = value.view(num_edges, self.heads, self.head_dim)
        scores = (query * key).sum(dim=-1) * self.scale

        trust = self.trust(historical_embeddings_i,historical_embeddings_j)
        scores = ( scores * trust.unsqueeze(-1))

        attention = torch.stack([softmax(scores[:, head], index)
                                     for head in range(self.heads)],dim=1)

        attention = self.dropout(attention)
        message = (value * attention.unsqueeze(-1))
        message = message.reshape(num_edges,self.dim)
        message = self.message_rope(message,edge_t)

        return self.output_projection(message)
