import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from .anomaly_utils import cosine_similarity


## Multi-Level Contrastive Learning Framework
class MultiLevelContrastiveLoss(nn.Module):
    """
    Multi-level contrastive loss module (temporal +  structural).

    """

    def __init__(
        self,
        num_nodes: int,
        config,
        device: str = "cpu",
    ):
        super().__init__()
        self.num_nodes     = num_nodes
        self.dimension     = config.MODEL.HIDDEN_CHANNELS
        self.temperature   = config.TRAIN.LOSS.TAU
        self.ema_decay     = config.TRAIN.EM_DECAY
        self.num_negatives = config.TRAIN.LOSS.NUM_NEGATIVES
        self.device        = device
        self.error = config.TRAIN.ERROR if config.TRAIN.ERROR else 0.0

        self.register_buffer(
            "prototypes",
            F.normalize(torch.randn(num_nodes, self.dimension), dim=-1),
        )

    #  Levels 1  – Temporal 
    def temporal_infonce(
        self,
        h_current: torch.Tensor,
        prev_state: torch.Tensor,   # [N, D]  same nodes' memory one step ago (detached)
        node_idxs: torch.Tensor,
        w1: float = 1.0,
        w3: float = 0.5,
    ):
        """
        Temporal InfoNCE (level 1) 

        Level 1 — positive: same node one step ago (prev_state, from the
        Memory module), negatives: every other node one step ago (in-batch).

        """
        N = h_current.size(0)
        q = F.normalize(h_current, dim=-1)  # shared query, live gradient

        # ---- Level 1: temporal ----
        logits1 = cosine_similarity(h_current, prev_state) / self.temperature
        labels1 = torch.arange(N, device=q.device)
        l1 = F.cross_entropy(logits1, labels1)


        k_pos = self.prototypes[node_idxs].detach()                     # [N, D]
        K = min(self.num_negatives, self.num_nodes)
        neg_idx = torch.randint(0, self.num_nodes, (K,), device=q.device)
        k_neg = self.prototypes[neg_idx].detach()                       # [K, D]

        pos_logit = (q * k_pos).sum(dim=-1, keepdim=True) / self.temperature
        neg_logit = cosine_similarity(h_current, k_neg) / self.temperature

        logits3 = torch.cat([pos_logit, neg_logit], dim=-1)             # [N, 1+K]
        labels3 = torch.zeros(N, dtype=torch.long, device=q.device)
        l3 = F.cross_entropy(logits3, labels3)

        # EMA prototype update (no grad flows through this)
        with torch.no_grad():
            alpha = self.ema_decay
            updated = alpha * self.prototypes[node_idxs] + (1 - alpha) * q
            self.prototypes[node_idxs] = F.normalize(updated, dim=-1)

        loss = w1 * l1 + w3 * l3
        return loss

    #  Level 2 – Structural InfoNCE
    def structural_infonce(
        self,
        h_current: torch.Tensor,
        edge_index: torch.Tensor,
        node_idxs: torch.Tensor,
    ) -> torch.Tensor:
        """
        InfoNCE over graph structure.

        Positive: mean-pool of the node's 1-hop neighbours.
        Negatives: all non-neighbour nodes in the batch (in-batch).

        A node should be more similar to its neighbourhood than to random
        nodes.
        """
        N = h_current.size(0)
        if edge_index is None or edge_index.size(1) == 0:
            return h_current.new_zeros(1).squeeze()

        idx_map = {nid.item(): i for i, nid in enumerate(node_idxs)}

        neigh_agg = torch.zeros_like(h_current)   # [N, D]
        neigh_cnt = torch.zeros(N, device=h_current.device)
        for s_global, d_global in zip(
            edge_index[0].tolist(), edge_index[1].tolist()
        ):
            if s_global in idx_map and d_global in idx_map:
                i_s = idx_map[s_global]
                i_d = idx_map[d_global]
                neigh_agg[i_s] = neigh_agg[i_s] + h_current[i_d].detach()
                neigh_cnt[i_s] += 1

        mask = neigh_cnt > 0
        if mask.sum() < 2:          # need at least 2 for in-batch negatives
            return h_current.new_zeros(1).squeeze()

        neigh_agg[mask] = neigh_agg[mask] / neigh_cnt[mask].unsqueeze(-1)
        logits = cosine_similarity(h_current[mask], neigh_agg[mask]) / self.temperature
        labels = torch.arange(mask.sum().item(), device=h_current.device)
        return F.cross_entropy(logits, labels)

    #  Combined final loss
    def forward(
        self,
        h_current: torch.Tensor,
        prev_state: torch.Tensor,
        node_idxs: torch.Tensor,
        edge_index: torch.Tensor,
        w1: float = 1.0,
        w2: float = 1.0,
        w3: float = 0.5,
    ):
        h_current = h_current + self.error * torch.randn_like(h_current)

        l1 = self.temporal_infonce(h_current, prev_state, node_idxs, w1=w1, w3=w3)
        l2 = self.structural_infonce(h_current, edge_index, node_idxs)

        loss = w1 * l1 + w2 * l2
        return loss