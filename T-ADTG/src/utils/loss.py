import torch
import torch.nn.functional as F


## Cosine Similarity Baseline implimentation
def cosine_similarity(z1, z2):
    assert z1.size() == z2.size()
    z1 = F.normalize(z1, p=2, dim=1)
    z2 = F.normalize(z2, p=2, dim=1)
    return torch.mm(z1, z2.t())

## Contrastive Loss Baseline 
def contrastive_loss(a, b, tau=0.02):
    similarity = cosine_similarity(a, b) / tau
    labels = torch.arange(
        similarity.size(0), dtype=torch.long, device=similarity.device
    )
    return F.cross_entropy(similarity, labels)

