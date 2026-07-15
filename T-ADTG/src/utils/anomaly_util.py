"""Temporal anomaly scoring from the node history window."""

import torch
import torch.nn.functional as F


def weighted_anomaly_score(history_deque, nodes_idx, decay=1.0):
    """Recency-weighted mean of cosine drift across consecutive history states.

    For each node, computes 1 - cos(h_t, h_{t-1}) over the history window and
    averages the steps with weights decay^t, so decay > 1 emphasises the most
    recent transitions. Higher values indicate more anomalous behaviour.
    """
    history_list = list(history_deque)
    T = len(history_list)

    scores, weights = [], []
    for t in range(1, T):
        h_t = history_list[t][nodes_idx]
        h_prev = history_list[t - 1][nodes_idx]
        scores.append(1 - F.cosine_similarity(h_t, h_prev, dim=-1))
        weights.append(decay ** t)

    scores = torch.stack(scores, dim=0)  # [T-1, n_nodes]
    w = torch.tensor(weights, dtype=scores.dtype, device=scores.device)
    w = w / w.sum()

    return (scores * w.unsqueeze(1)).sum(dim=0)  # [n_nodes]