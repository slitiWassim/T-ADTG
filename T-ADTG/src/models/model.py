"""T-ADTG encoder: stateless node encoder for anomaly detection on CTDGs.

Produces node representations consumed by the multi-level contrastive
objective. Temporal memory lives in the external History module, so the
encoder itself is a pure function of the current sub-graph and the learned
identity table.
"""

from typing import Optional

import torch
import torch.nn as nn
from torch_geometric.utils import degree

from layers import TrustTemporalAttention
from time_encoding import TimeEncoder


class TADTG(nn.Module):
    """Trust-aware temporal graph encoder.

    Expected batch fields (from the EventLoader / Collater):
        x           (N, in_dim)    node features (all-zero on Wiki/BTC/Amazon)
        n_id        (N,)           local -> global node id map
        edge_index  (2, E)         local directed edges [root, neighbour]
        src, dst    (E,)           local endpoints
        msg         (E, edge_dim)  edge features
        t           (E,)           edge timestamps
        batch_size                 number of root events (roots come first)

    Returns node representations (N, hidden_dim); the readout for root b is
    x[batch.src[b]]. Per-edge trust from the last forward pass is available
    through get_trust_scores(), aligned to edge_index columns.
    """

    def __init__(self,
                 num_nodes: int,
                 in_dim: int,
                 edge_dim: int,
                 hidden_dim: int,
                 num_layers: int = 2,
                 heads: int = 8,
                 dropout: float = 0.1,
                 use_input_feat: Optional[bool] = None,
                 **kwargs):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_nodes = num_nodes

        # On zero-feature datasets identity comes from the embedding table;
        # projected input features are only added when they carry information.
        if use_input_feat is None:
            use_input_feat = in_dim is not None and in_dim > 1
        self.use_input_feat = bool(use_input_feat)

        self.node_emb = nn.Embedding(num_nodes, hidden_dim)
        self.input_proj = nn.Linear(in_dim, hidden_dim) if self.use_input_feat else None
        self.edge_enc = nn.Linear(edge_dim, hidden_dim)
        self.time_enc = TimeEncoder(hidden_dim)

        self.layers = nn.ModuleList([
            TrustTemporalAttention(hidden_dim, heads=heads, dropout=dropout)
            for _ in range(num_layers)
        ])
        self.final_norm = nn.LayerNorm(hidden_dim)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        nn.init.normal_(self.node_emb.weight, std=0.1)

    def forward(self, batch) -> torch.Tensor:
        N = batch.x.size(0)
        device = batch.x.device
        edge_index = batch.edge_index
        src, dst = edge_index
        t = batch.t.float()

        x = self.node_emb(batch.n_id)
        if self.use_input_feat:
            x = x + self.input_proj(batch.x)

        edge_emb = self.edge_enc(batch.msg)

        deg = (degree(src, N, dtype=torch.float)
               + degree(dst, N, dtype=torch.float)).clamp(min=1.0)

        # Relative time = age of each edge w.r.t. the source node's most
        # recent event in this sub-graph; non-negative because sampled
        # neighbours satisfy t_edge <= t_root.
        node_last_t = torch.zeros(N, device=device)
        node_last_t.scatter_reduce_(0, src, t, reduce="amax", include_self=True)
        t_rel = (node_last_t[src] - t).clamp(min=0.0)
        time_emb = self.time_enc(t, t_rel)
        t_feat = (t_rel - t_rel.mean()) / (t_rel.std() + 1e-6)

        for layer in self.layers:
            x = layer(x, edge_index, edge_emb, time_emb, t_feat, deg)

        return self.final_norm(x)

    def get_trust_scores(self, layer_idx: int = -1) -> Optional[torch.Tensor]:
        """Per-edge trust (E,) in (0, 2) cached during the last forward pass."""
        return self.layers[layer_idx]._trust_cache

    @torch.no_grad()
    def reset_memory(self):
        """No-op kept for API symmetry: all resettable state is in History."""
        return