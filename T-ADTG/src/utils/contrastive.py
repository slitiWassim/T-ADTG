"""Multi-level contrastive objective and structural anomaly score for T-ADTG."""

from collections import deque

import torch
import torch.nn as nn
import torch.nn.functional as F


def _neighbour_mean(h: torch.Tensor, edge_index: torch.Tensor):
    """Mean-pool neighbour embeddings into each root node.

    Edges follow the sampler convention [root, neighbour]; the aggregate for
    a root is the mean of its neighbours' embeddings. Returns the pooled
    tensor and a mask of nodes with at least one neighbour.
    """
    N = h.size(0)
    src, dst = edge_index

    agg = torch.zeros_like(h)
    agg.index_add_(0, src, h[dst])
    cnt = torch.zeros(N, device=h.device)
    cnt.index_add_(0, src, torch.ones_like(src, dtype=torch.float))

    mask = cnt > 0
    agg[mask] = agg[mask] / cnt[mask].unsqueeze(-1)
    return agg, mask


@torch.no_grad()
def structural_anomaly_score(h: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
    """Cosine distance between each node and its neighbourhood mean.

    Nodes without neighbours in the sub-graph score zero.
    """
    score = torch.zeros(h.size(0), device=h.device)
    if edge_index is None or edge_index.size(1) == 0:
        return score

    agg, mask = _neighbour_mean(h, edge_index)
    cos = F.cosine_similarity(h[mask], agg[mask], dim=-1)
    score[mask] = (1.0 - cos).clamp(min=0)
    return score


class MultiLevelContrastiveLoss(nn.Module):
    """Temporal (L1) and structural (L2) InfoNCE losses.

    L1 contrasts a node's current embedding against its own state one history
    step earlier, with in-batch negatives. L2 contrasts a node against the
    mean of its 1-hop neighbourhood in the sampled sub-graph, again with
    in-batch negatives. Both follow the standard NT-Xent formulation.
    """

    def __init__(
        self,
        num_nodes: int,
        dimension: int,
        temperature: float = 0.07,
        ema_decay: float = 0.99,      # kept for config compatibility, unused
        num_negatives: int = 64,      # kept for config compatibility, unused
        device: str = "cpu",
    ):
        super().__init__()
        self.num_nodes = num_nodes
        self.dimension = dimension
        self.temperature = temperature
        self.device = device

    def temporal_infonce(
        self,
        h_current: torch.Tensor,   # [B, D] root embeddings at time t
        history_deque: deque,      # history slots, each [num_nodes, D]
        node_idxs: torch.Tensor,   # [B] global root ids
    ) -> torch.Tensor:
        history_list = list(history_deque)
        if len(history_list) < 2:
            return h_current.new_zeros(1).squeeze()

        q = F.normalize(h_current, dim=-1)
        # Key: same node one step earlier, detached so the objective cannot
        # be satisfied by dragging the past towards the present.
        k = F.normalize(history_list[-2][node_idxs].detach(), dim=-1)

        logits = torch.mm(q, k.T) / self.temperature
        labels = torch.arange(q.size(0), device=q.device)
        return F.cross_entropy(logits, labels)

    def structural_infonce(
        self,
        h_graph: torch.Tensor,     # [N, D] embeddings of ALL sampled nodes
        edge_index: torch.Tensor,  # [2, E] LOCAL edges [root, neighbour]
    ) -> torch.Tensor:
        if edge_index is None or edge_index.size(1) == 0:
            return h_graph.new_zeros(1).squeeze()

        # Neighbour aggregate is detached: the trivial solution of collapsing
        # onto the neighbourhood mean should not be reachable by gradient.
        agg, mask = _neighbour_mean(h_graph.detach(), edge_index)
        if mask.sum() < 2:  # need at least two anchors for in-batch negatives
            return h_graph.new_zeros(1).squeeze()

        q = F.normalize(h_graph[mask], dim=-1)
        k = F.normalize(agg[mask], dim=-1)

        logits = torch.mm(q, k.T) / self.temperature
        labels = torch.arange(q.size(0), device=q.device)
        return F.cross_entropy(logits, labels)

    def forward(
        self,
        h_current: torch.Tensor,        # [B, D] root embeddings after History
        h_graph: torch.Tensor,          # [N, D] all sampled node embeddings
        history_deque: deque,
        node_idxs: torch.Tensor,        # [B] global root ids
        sampled_node_ids: torch.Tensor, # [N] local -> global map (kept in
                                        # the signature for API stability)
        edge_index: torch.Tensor,       # [2, E] LOCAL edges
        w1: float = 1.0,
        w2: float = 0.5,
        w3: float = 0.0,
    ):
        l1 = self.temporal_infonce(h_current, history_deque, node_idxs)
        l2 = self.structural_infonce(h_graph, edge_index)

        loss = w1 * l1 + w2 * l2
        detail = {
            "L1_temporal": float(l1.item()),
            "L2_structural": float(l2.item()),
            "L3_prototype": 0.0,
        }
        return loss, detail