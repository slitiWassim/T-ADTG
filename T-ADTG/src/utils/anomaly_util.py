import math
import numpy as np
from sklearn import metrics
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import deque
from .contrastive import MultiLevelContrastiveLoss 
from .loss import contrastive_loss , cosine_similarity

# ============================================================================
#  Multi-level anomaly score
# ============================================================================

@torch.no_grad()
def multi_level_anomaly_score(
    h_current: torch.Tensor,                         # [N, D]
    history_deque: deque,
    node_idxs: torch.Tensor,                         # [N]  global IDs
    edge_index: torch.Tensor,                        # [2, E]
    contrastive_module: MultiLevelContrastiveLoss,
    alpha: float = 0.40,   # weight for temporal component
    beta:  float = 0.30,   # weight for structural component
    gamma: float = 0.30,   # weight for prototype component
) -> torch.Tensor:
    """
    Compute the three-component anomaly score for each node.

    S1 – temporal drift
        Measures how much a node's embedding has changed on average across the
        history window. A 2nd-order "acceleration" term is added: a sudden
        *change in rate of change* is a stronger anomaly signal than a steady
        drift.

    S2 – structural outlier
        How far the node's current embedding is from its neighbourhood
        aggregate. A node that starts connecting to very different peers, or
        whose neighbours' representations change abruptly, will score high.

    S3 – prototype outlier
        Cosine distance from the node's own EMA prototype. The prototype
        accumulates a stable "normal" baseline; large deviations flag anomalies
        without any supervised labels.

    Returns
    -------
    score : [N] tensor, higher = more anomalous
    """
    N            = h_current.size(0)
    history_list = list(history_deque)

    # ── S1: temporal drift with 2nd-order acceleration ─────────────────
    s1 = torch.zeros(N, device=h_current.device)
    if len(history_list) >= 2:
        cos_series = []
        for t in range(1, len(history_list)):
            h_t    = history_list[t][node_idxs]
            h_prev = history_list[t - 1][node_idxs]
            cos    = F.cosine_similarity(h_t, h_prev, dim=-1)
            s1    += (1.0 - cos)
            cos_series.append(cos)
        s1 /= (len(history_list) - 1)   # mean drift

        # 2nd-order: variance in the drift signal catches sudden spikes
        s1_accel = torch.zeros(N, device=h_current.device)
        if len(cos_series) >= 2:
            for i in range(1, len(cos_series)):
                s1_accel += (cos_series[i] - cos_series[i - 1]).abs()
            s1_accel /= (len(cos_series) - 1)

        s1 = 0.70 * s1 + 0.30 * s1_accel

    # ── S2: structural outlier ─────────────────────────────────────────
    s2 = torch.zeros(N, device=h_current.device)
    if edge_index is not None and edge_index.size(1) > 0:
        idx_map   = {nid.item(): i for i, nid in enumerate(node_idxs)}
        neigh_agg = torch.zeros_like(h_current)
        neigh_cnt = torch.zeros(N, device=h_current.device)

        for s_g, d_g in zip(edge_index[0].tolist(), edge_index[1].tolist()):
            if s_g in idx_map and d_g in idx_map:
                i_s = idx_map[s_g]
                i_d = idx_map[d_g]
                neigh_agg[i_s] = neigh_agg[i_s] + h_current[i_d]
                neigh_cnt[i_s] += 1

        mask = neigh_cnt > 0
        if mask.sum() > 0:
            neigh_agg[mask] = neigh_agg[mask] / neigh_cnt[mask].unsqueeze(-1)
            cos_struct       = F.cosine_similarity(h_current, neigh_agg, dim=-1)
            s2[mask]         = (1.0 - cos_struct[mask]).clamp(min=0)

    # ── S3: prototype outlier ──────────────────────────────────────────
    proto = contrastive_module.prototypes[node_idxs]         # [N, D]
    s3    = (1.0 - F.cosine_similarity(h_current, proto, dim=-1)).clamp(min=0)

    # ── Weighted combination ───────────────────────────────────────────
    score = alpha * s1 + beta * s2 + gamma * s3
    return score   # [N], higher → more anomalous



#################### Anomaly Score ###############
def temporal_anomaly_score(history_deque, nodes_idx):

    history_list = list(history_deque)

    scores = []

    for t in range(1, len(history_list)):

        h_t = history_list[t][nodes_idx]
        h_prev = history_list[t-1][nodes_idx]

        cos = F.cosine_similarity(h_t, h_prev, dim=-1)

        scores.append(1 - cos)   # anomaly signal

    scores = torch.stack(scores, dim=0)  # [T-1, n_nodes]

    return scores.mean(dim=0)            # [n_nodes]





#### Anomaly Score : Claude proposed  anomaly score to be tested
def anomaly_score(history_deque, nodes_idx,
                           decay=2.0, alpha=0.6, beta=0.3):
    """
    Improved temporal anomaly scoring with:
      - Exponential recency weighting : recent transitions dominate
      - Hybrid metric                 : cosine dissimilarity + normalised L2
      - Max-mean blend                : catches abrupt spikes AND sustained drift
      - Long-range anchor             : latest state vs. oldest (global drift)

    Args:
        history_deque : deque of hidden-state tensors
        nodes_idx     : node index selector
        decay (float) : exponential base for recency weights (>1 → recent matters more)
                        try values in [1.5, 3.0]
        alpha (float) : cosine weight vs L2 weight          [0, 1]
        beta  (float) : blend of max-score into weighted-mean [0, 1]

    Returns:
        anomaly scores  [n_nodes]   (higher = more anomalous)
    """
    history_list = list(history_deque)
    T = len(history_list)

    if T < 2:
        h = history_list[-1][nodes_idx]
        return torch.zeros(h.shape[:-1], device=h.device, dtype=h.dtype)

    step_scores = []
    weights     = []

    for t in range(1, T):
        h_t    = history_list[t    ][nodes_idx]   # [n_nodes, d]
        h_prev = history_list[t - 1][nodes_idx]   # [n_nodes, d]

        # ── 1. Cosine dissimilarity ────────────────────────────────────────
        cos_score = 1.0 - F.cosine_similarity(h_t, h_prev, dim=-1)  # ∈ [0, 2]

        # ── 2. Scale-invariant L2 distance ────────────────────────────────
        l2       = torch.norm(h_t - h_prev, dim=-1)
        avg_norm = (torch.norm(h_t, dim=-1) + torch.norm(h_prev, dim=-1)) * 0.5 + 1e-8
        l2_score = l2 / avg_norm

        # ── 3. Combined step score ─────────────────────────────────────────
        step = alpha * cos_score + (1.0 - alpha) * l2_score
        step_scores.append(step)

        # Exponential recency weight: t=1 → 1, t=2 → decay, …, t=T-1 → decay^(T-2)
        weights.append(decay ** (t - 1))

    step_scores = torch.stack(step_scores, dim=0)       # [T-1, n_nodes]

    # ── 4. Recency-weighted mean ───────────────────────────────────────────
    w = torch.tensor(weights, dtype=step_scores.dtype, device=step_scores.device)
    w = w / w.sum()
    weighted_mean = (step_scores * w.unsqueeze(1)).sum(dim=0)   # [n_nodes]

    # ── 5. Max score  (catches abrupt isolated spikes) ────────────────────
    max_score, _ = step_scores.max(dim=0)                        # [n_nodes]

    # ── 6. Long-range anchor: latest vs. oldest (global drift) ───────────
    h_last, h_first = history_list[-1][nodes_idx], history_list[0][nodes_idx]
    anchor_cos  = 1.0 - F.cosine_similarity(h_last, h_first, dim=-1)
    anchor_norm = (torch.norm(h_last, dim=-1) + torch.norm(h_first, dim=-1)) * 0.5 + 1e-8
    anchor_l2   = torch.norm(h_last - h_first, dim=-1) / anchor_norm
    anchor      = alpha * anchor_cos + (1.0 - alpha) * anchor_l2   # [n_nodes]

    # ── 7. Final blend ─────────────────────────────────────────────────────
    #  (1-β)·weighted_mean + β·max  →  then mix in global anchor
    step_blend = (1.0 - beta) * weighted_mean + beta * max_score
    final      = 0.75 * step_blend + 0.25 * anchor

    return final                                                    # [n_nodes]



def weighted_anomaly_score(history_deque, nodes_idx, decay=1.0):
    history_list = list(history_deque)
    T = len(history_list)

    scores  = []
    weights = []

    for t in range(1, T):
        h_t    = history_list[t    ][nodes_idx]
        h_prev = history_list[t - 1][nodes_idx]

        cos = F.cosine_similarity(h_t, h_prev, dim=-1)
        scores.append(1 - cos)

        # most recent transition gets weight decay^(T-1), oldest gets decay^1
        weights.append(decay ** t)

    scores  = torch.stack(scores, dim=0)                          # [T-1, n_nodes]
    w       = torch.tensor(weights, dtype=scores.dtype,
                           device=scores.device)
    w       = w / w.sum()                                         # normalise

    return (scores * w.unsqueeze(1)).sum(dim=0)                   # [n_nodes]



### Same Anomaly Score but modified by : Wassim Sliti

#################### Anomaly Score ###############
def anomaly_score_cosine_similarity(history_deque, nodes_idx):

    history_list = list(history_deque)

    scores = []

    for t in range(1, len(history_list)):

        h_t = history_list[t][nodes_idx]
        h_prev = history_list[t-1][nodes_idx]

        #cos = F.cosine_similarity(h_t, h_prev, dim=-1)
        cos =  1-torch.diag(cosine_similarity(h_t, h_prev)).view(-1)
        scores.append(cos)   # anomaly signal

    scores = torch.stack(scores, dim=0)  # [T-1, n_nodes]

    return scores.mean(dim=0)            # [n_nodes]