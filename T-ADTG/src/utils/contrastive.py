import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import deque
from tqdm import tqdm



# ============================================================================
#  Multi-level contrastive framework
# ============================================================================
class MultiLevelContrastiveLoss(nn.Module):
    """
    Three-level contrastive loss module.

    Parameters
    ----------
    num_nodes    : total nodes in the graph (for prototype bank)
    dimension    : embedding dimension
    temperature  : softmax temperature (default 0.07 – standard for InfoNCE)
    ema_decay    : EMA coefficient for prototype update (0.99 is a safe default)
    num_negatives: number of random prototype negatives sampled per forward pass
    device       : torch device
    """

    def __init__(
        self,
        num_nodes: int,
        dimension: int,
        temperature: float = 0.07,
        ema_decay: float = 0.99,
        num_negatives: int = 64,
        device: str = "cpu",
    ):
        super().__init__()
        self.num_nodes     = num_nodes
        self.dimension     = dimension
        self.temperature   = temperature
        self.ema_decay     = ema_decay
        self.num_negatives = num_negatives
        self.device        = device

        # EMA prototype bank – unit-normalised, one row per node.
        # Registered as a buffer so it moves with .to(device) but is NOT a
        # learnable parameter (gradients should NOT flow through it).
        self.register_buffer(
            "prototypes",
            F.normalize(torch.randn(num_nodes, dimension), dim=-1),
        )

    # ------------------------------------------------------------------ #
    #  Level 1 – Temporal InfoNCE                                          #
    # ------------------------------------------------------------------ #
    def temporal_infonce(
        self,
        h_current: torch.Tensor,   # [N, D]  embeddings at time t
        history_deque: deque,      # full history deque from History module
        node_idxs: torch.Tensor,   # [N]     global node IDs
    ) -> torch.Tensor:
        """
        InfoNCE over time.

        Positive: same node one step ago.
        Negatives: every OTHER node one step ago (in-batch negatives).

        The [N x N] logit matrix has the diagonal as positives – exactly the
        standard NT-Xent / SimCLR setup. The model must learn to match a node
        to its own temporal past rather than any other node's past.
        """
        history_list = list(history_deque)
        if len(history_list) < 2:
            return h_current.new_zeros(1).squeeze()

        # Query: current embeddings (live gradient)
        q = F.normalize(h_current, dim=-1)                              # [N, D]

        # Key: same node at t-1, detached (stable target).
        # Detach prevents the "push the past to match the present" shortcut.
        k = F.normalize(
            history_list[-2][node_idxs].detach(), dim=-1
        )                                                                # [N, D]

        # Similarity matrix: q[i] · k[j] for all (i,j) pairs
        logits = torch.mm(q, k.T) / self.temperature                    # [N, N]

        # Positive = diagonal (same node identity)
        labels = torch.arange(q.size(0), device=q.device)
        return F.cross_entropy(logits, labels)

    # ------------------------------------------------------------------ #
    #  Level 2 – Structural InfoNCE                                        #
    # ------------------------------------------------------------------ #
    def structural_infonce(
        self,
        h_current: torch.Tensor,   # [N, D]
        edge_index: torch.Tensor,  # [2, E]  edges in the current subgraph
        node_idxs: torch.Tensor,   # [N]     global node IDs in this batch
    ) -> torch.Tensor:
        """
        InfoNCE over graph structure.

        Positive: mean-pool of the node's 1-hop neighbours.
        Negatives: all non-neighbour nodes in the batch (in-batch).

        A node should be more similar to its neighbourhood than to random
        nodes. Anomalous nodes that suddenly change their neighbourhood
        pattern will get a high structural score.
        """
        N = h_current.size(0)
        K       = min(self.num_negatives, self.num_nodes)
        neg_idx = torch.randint(0, self.num_nodes, (K,), device=h_current.device)
        if edge_index is None or edge_index.size(1) == 0:
            return h_current.new_zeros(1).squeeze()

        # Map global node IDs to local [0..N-1] indices for fast lookup
        idx_map = {nid.item(): i for i, nid in enumerate(node_idxs)}

        # Aggregate neighbour embeddings (mean pooling, detached to avoid
        # trivial solution of "just become your neighbours")
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

        # Only compute loss for nodes that have at least one neighbour
        mask = neigh_cnt > 0
        if mask.sum() < 2:          # need at least 2 for in-batch negatives
            return h_current.new_zeros(1).squeeze()

        neigh_agg[mask] = neigh_agg[mask] / neigh_cnt[mask].unsqueeze(-1)

        q = F.normalize(h_current[mask], dim=-1)    # [M, D]  anchor
        k = F.normalize(neigh_agg[mask], dim=-1)    # [M, D]  positive key

        # [M, M] logit matrix – diagonal is the positive pair
        logits = torch.mm(q, k.T) / self.temperature
        labels = torch.arange(q.size(0), device=q.device)
        return F.cross_entropy(logits, labels)

    # ------------------------------------------------------------------ #
    #  Level 3 – Prototype InfoNCE                                         #
    # ------------------------------------------------------------------ #
    def prototype_infonce(
        self,
        h_current: torch.Tensor,   # [N, D]
        node_idxs: torch.Tensor,   # [N]     global node IDs
    ) -> torch.Tensor:
        """
        InfoNCE against the EMA prototype bank.

        Positive: the node's own EMA prototype (its historical "normal" state).
        Negatives: K randomly sampled prototypes from other nodes.

        This level provides a *long-term* memory signal: the model is penalised
        when a node's current embedding drifts far from its accumulated history,
        regardless of what happened in the current mini-batch.

        The prototype bank is updated via EMA after computing the loss so the
        update does not interfere with gradient computation.
        """
        N  = h_current.size(0)
        q  = F.normalize(h_current, dim=-1)          # [N, D]  live gradient

        # Positive key: node's own prototype (no gradient – bank is a buffer)
        k_pos = self.prototypes[node_idxs].detach()  # [N, D]

        # Negative keys: random sample from prototype bank
        K       = min(self.num_negatives, self.num_nodes)
        neg_idx = torch.randint(0, self.num_nodes, (K,), device=q.device)
        k_neg   = self.prototypes[neg_idx].detach()  # [K, D]

        # [N, 1]  positive logit per anchor
        pos_logit = (q * k_pos).sum(dim=-1, keepdim=True) / self.temperature

        # [N, K]  negative logits per anchor
        neg_logit = torch.mm(q, k_neg.T) / self.temperature

        # Concatenate: column 0 is always the positive
        logits = torch.cat([pos_logit, neg_logit], dim=-1)          # [N, 1+K]
        labels = torch.zeros(N, dtype=torch.long, device=q.device)  # all 0

        loss = F.cross_entropy(logits, labels)

        # EMA prototype update (no grad flows through this)
        with torch.no_grad():
            alpha = self.ema_decay
            updated = alpha * self.prototypes[node_idxs] + (1 - alpha) * q
            self.prototypes[node_idxs] = F.normalize(updated, dim=-1)

        return loss

    # ------------------------------------------------------------------ #
    #  Combined forward                                                    #
    # ------------------------------------------------------------------ #
    def forward(
        self,
        h_current: torch.Tensor,
        history_deque: deque,
        node_idxs: torch.Tensor,
        edge_index: torch.Tensor,
        w1: float = 1.0,
        w2: float = 0.5,
        w3: float = 0.5,
    ):
        """
        Returns
        -------
        loss   : scalar tensor (differentiable)
        detail : dict with per-level loss values for logging
        """
        l1 = self.temporal_infonce(h_current, history_deque, node_idxs)
        l2 = self.structural_infonce(h_current, edge_index, node_idxs)
        #l3 = self.prototype_infonce(h_current, node_idxs)

        loss = w1 * l1 + w2 * l2 
        detail = {
            "L1_temporal":    l1.item(),
            "L2_structural":  l2.item(),
            "L3_prototype":   0.0,
        }
        return loss, detail




# ============================================================================
#  The sum of the loss of every pairs of consecutive pairs of node representation
# ============================================================================

def sce_temporal_loss(history_deque, nodes_idx):
    history_list = list(history_deque)
    loss = None

    for t in range(1, len(history_list)):
        h_t   = history_list[t][nodes_idx]            # history[-1]: live grad ✓
        h_prev = history_list[t-1][nodes_idx].detach() # older slots: anchor only
        #h_t = F.normalize(h_t, dim=-1)
        #h_prev = F.normalize(h_prev, dim=-1)
        step_loss = (1 - F.cosine_similarity(h_t, h_prev, dim=-1)).mean()
        loss = step_loss if loss is None else loss + step_loss

    return loss / (len(history_list) - 1)



#####################################################################################
# The sum of the loss of the curr state and each saved historical node representation
#####################################################################################

def sce_temporal_loss_cur(cur_emb, history_deque, nodes_idx):
    """
    cur_emb     : current GRU output WITH grad_fn  [batch, dim]
    history_deque: deque of detached tensors        [num_nodes, dim] each slot
    nodes_idx   : indices of the current batch nodes

    Compares cur_emb against every previous history slot.
    Gradient flows through cur_emb → model params. ✓
    """
    loss = None
    for slot in history_deque:
        h_prev    = slot[nodes_idx].detach()     # anchor, no grad
        step_loss = (1 - F.cosine_similarity(cur_emb, h_prev, dim=-1)).mean()
        loss      = step_loss if loss is None else loss + step_loss
    return loss / len(history_deque)




############### Loss Tracking the loss of each consecutive representation in the history with negative sampling
###############################################################################################################

def temporal_history_contrastive_loss(
        history,
        nodes_idx,
        temperature=0.2,
        num_neg=50
):
    
    history_list = list(history.history)
    K = len(history_list)

    device = history_list[-1].device
    h_current = history_list[-1][nodes_idx]

    h_current = F.normalize(h_current, dim=-1)

    positives = []
    negatives = []

    # iterate over full history
    for k in range(K-1):

        h_hist = history_list[k].detach()
        h_hist = F.normalize(h_hist, dim=-1)

        # positive: same node
        positives.append(h_hist[nodes_idx])

        # negative sampling
        rand_idx = torch.randint(
            0, history.num_nodes,
            (len(nodes_idx), num_neg),
            device=device
        )

        negatives.append(h_hist[rand_idx])

    positives = torch.stack(positives, dim=1)        # [B,K-1,D]
    negatives = torch.stack(negatives, dim=1)        # [B,K-1,N,D]

    # similarity with positives
    pos_sim = torch.einsum("bd,bkd->bk", h_current, positives) / temperature

    # similarity with negatives
    neg_sim = torch.einsum("bd,bknd->bkn", h_current, negatives) / temperature

    logits = torch.cat(
        [pos_sim.unsqueeze(-1), neg_sim],
        dim=-1
    )  # [B,K-1,N+1]

    logits = logits.view(-1, num_neg + 1)

    labels = torch.zeros(
        logits.shape[0],
        dtype=torch.long,
        device=device
    )

    loss = F.cross_entropy(logits, labels)

    return loss



