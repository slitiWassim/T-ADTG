import os
import random

import numpy as np
import torch
from tqdm import tqdm
from src.utils.anomaly_utils import anomaly_score



def set_seed(seed: int):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.enabled = True  



def train(loader, model, memory, contrastive_module, optimizer, device, data, config):
    """Runs one training epoch: encodes each batch, updates the temporal
    memory, and optimizes the multi-level contrastive loss."""
    model.train()
    memory.train()
    contrastive_module.train()

    pbar = tqdm(loader)
    total_loss = 0.0

    for batch in pbar:
        batch = batch.to(device)
        optimizer.zero_grad()

        out = model(batch)
        src = batch.src[: batch.batch_size]
        out = out[src]                              

        idx = batch.input_id.cpu()
        cur_mem, prev_mem = memory(out, data.src[idx])     

        loss = contrastive_module(
            h_current=cur_mem,
            prev_state = prev_mem,
            node_idxs=data.src[idx],
            edge_index=batch.edge_index,
            w1= config.TRAIN.LOSS.W1,
            w2= config.TRAIN.LOSS.W2,
            w3= config.TRAIN.LOSS.W3)

        loss.backward()
        # Gradient clipping prevents exploding gradients when InfoNCE loss
        # landscapes become steep near the boundary of a good representation
        torch.nn.utils.clip_grad_norm_(
            list(model.parameters()) + list(memory.parameters()), max_norm=1.0
        )
        optimizer.step()

        total_loss += loss.item()
        pbar.set_description(
            f"Loss={loss.item():.4f} "
        )

    n = len(loader)
    return total_loss / n 


@torch.no_grad()
def inference(loader, model, memory, contrastive_module, metrics, device, data, config):
    """Scores every event in `loader` for anomaly and reports AUROC/AUPRC.

    Combines a prediction-drift score (change between the current and
    previous memory state) with a structural anomaly score, then evaluates
    against ground-truth labels (excluding the "unknown" class, value 2).
    """
    preds, labels = [], []
    memory.eval()
    contrastive_module.eval()

    for batch in tqdm(loader):
        batch = batch.to(device)

        out = model(batch)
        src = batch.src[: batch.batch_size]
        out = out[src]
        label = batch.y[: batch.batch_size]

        idx = batch.input_id.cpu()
        cur_mem, prev_mem = memory(out, data.src[idx])

        score = anomaly_score(cur_mem,
                              prev_mem,
                              batch.edge_index,
                              data.src[idx])

        torch.cuda.empty_cache()
        preds.append(score.sigmoid().cpu())
        labels.append(label.cpu())

    preds = torch.cat(preds)
    labels = torch.cat(labels)

    # Replace any NaN scores with 0
    preds[torch.isnan(preds)] = 0.0

    # Exclude "unknown" labels (value == 2)
    mask = labels != 2
    return metrics(labels[mask], preds[mask])
