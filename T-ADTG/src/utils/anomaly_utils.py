
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np



## Cosine Similarity Baseline implementation
def cosine_similarity(z1, z2):
    assert z1.size(-1) == z2.size(-1), (
        f"Feature dims must match, got {z1.size(-1)} and {z2.size(-1)}"
    )
    z1 = F.normalize(z1, p=2, dim=1)
    z2 = F.normalize(z2, p=2, dim=1)
    return torch.mm(z1, z2.t())
    

@torch.no_grad()
def temporal_deviation(
     h: torch.Tensor,
     prev_mem: torch.Tensor) -> torch.Tensor:
    return 1 - torch.diag(cosine_similarity(h, prev_mem)).view(-1)

@torch.no_grad()
def structural_deviation(
     h: torch.Tensor,
     edge_index: torch.Tensor,
     node_idxs: torch.Tensor) -> torch.Tensor:

    N     = h.size(0)
    score = torch.zeros(N, device=h.device)

    if edge_index is None or edge_index.size(1) == 0:
        return score

    idx_map   = {nid.item(): i for i, nid in enumerate(node_idxs)}
    neigh_agg = torch.zeros_like(h)
    neigh_cnt = torch.zeros(N, device=h.device)

    for s_g, d_g in zip(edge_index[0].tolist(), edge_index[1].tolist()):
        if s_g in idx_map and d_g in idx_map:
            i_s = idx_map[s_g]
            i_d = idx_map[d_g]
            neigh_agg[i_s] += h[i_d]
            neigh_cnt[i_s] += 1

    mask = neigh_cnt > 0
    if mask.any():
        score[mask] = 1 - torch.diag(cosine_similarity(h[mask], neigh_agg[mask])).view(-1)

    return score



@torch.no_grad()
def anomaly_score(h: torch.Tensor,
                  prev_mem: torch.Tensor,
                  edge_index: torch.Tensor,
                  node_idxs: torch.Tensor,
                  alpha: float = 1.00,
                  beta: float = 1.00) -> torch.Tensor:

    return alpha * temporal_deviation(h,prev_mem) + beta * structural_deviation(h,edge_index,node_idxs)
    
    