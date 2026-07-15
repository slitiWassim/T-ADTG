"""Training entry point for T-ADTG."""

import argparse
import logging
import os
import os.path as osp
import random

import mlflow
import numpy as np
import torch
from tqdm import tqdm

from src.config.defaults import _C as config, update_config, flatten_cfg
from src.datasets.dataset import load_dataset
from src.datasets.loader import train_val_test_load
from src.models.model import TADTG
from src.utils.contrastive_updated import (
    MultiLevelContrastiveLoss,
    structural_anomaly_score,
)
from src.utils.anomaly_util import weighted_anomaly_score
from src.utils.memory import Memory
from src.utils.loss import contrastive_loss, cosine_similarity
from src.utils.measure import Measure


def set_seed(seed: int):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def parse_args():
    parser = argparse.ArgumentParser(description="T-ADTG")
    parser.add_argument("--cfg", type=str, default="src/config/wikipedia.yaml",
                        help="experiment configuration filename")
    parser.add_argument("--gpu", type=int, default=0, help="GPU id")
    parser.add_argument("opts", nargs=argparse.REMAINDER, default=None,
                        help="modify config options from the command line")
    args = parser.parse_args()
    update_config(config, args)
    return args


def root_local_indices(root_global, n_id, num_nodes, device):
    """Map global root ids to their local positions within the sampled batch."""
    lookup = torch.full((num_nodes,), -1, dtype=torch.long, device=device)
    lookup[n_id] = torch.arange(n_id.size(0), device=device)
    return lookup[root_global]


def setup_logger(dataset_name):
    os.makedirs("log", exist_ok=True)
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(message)s",
        level=logging.INFO,
        filename=f"log/{osp.basename(__file__)[:-3]}_{dataset_name}.txt",
        force=True,
    )
    return logging.getLogger()


def main():
    set_seed(42)
    args = parse_args()

    device = torch.device(f"cuda:{args.gpu}")
    print(f"Using {device}")

    log = setup_logger(config.DATASET.NAME)
    log.info("########################## START #########################")
    log.info(f"\n{config.DATASET.NAME}-param: {flatten_cfg(config)}")

    data = load_dataset(config.DATASET.NAME, zero_edge=config.DATASET.ZERO_EDGE)
    train_loader, valid_loader, test_loader = train_val_test_load(data, config)

    measure = Measure("auc")
    best, best_val = 0.0, 0.0

    mlflow.set_experiment(config.DATASET.NAME)

    with mlflow.start_run(run_name=config.MODEL.NAME):
        mlflow.log_params(flatten_cfg(config))
        mlflow.set_tag("model", "TADTG")
        mlflow.set_tag(
            "experiment_description",
            "Temporal GNN with memory history and contrastive loss",
        )

        history = Memory(
            data.num_nodes,
            config.DATASET.NUM_TIMESLOTS,
            config.MODEL.HIDDEN_CHANNELS,
            device=device,
            recurrent=config.HISTORY.RECURRENT,
            normalize=config.HISTORY.NORMALIZE,
        ).to(device)

        model = TADTG(
            num_nodes=data.num_nodes,
            in_dim=data.x.size(1),
            edge_dim=data.msg.size(1),
            hidden_dim=config.MODEL.HIDDEN_CHANNELS,
            num_layers=config.MODEL.NUM_LAYERS,
            heads=config.MODEL.HEADS,
            dropout=config.MODEL.DROPOUT,
        ).to(device)

        print(model)
        mlflow.log_text(str(model), "model_architecture.txt")

        contrastive_module = MultiLevelContrastiveLoss(
            num_nodes=data.num_nodes,
            dimension=config.MODEL.HIDDEN_CHANNELS,
            temperature=config.TRAIN.TEMPRATURE,
            ema_decay=config.TRAIN.EM_DECAY,
            num_negatives=config.TRAIN.LOSS.NUM_NEGATIVES,
            device=device,
        ).to(device)

        optimizer = torch.optim.Adam(
            list(model.parameters())
            + list(history.parameters())
            + list(contrastive_module.parameters()),
            lr=config.TRAIN.LR,
            weight_decay=config.TRAIN.WD,  # small WD guards against InfoNCE collapse
        )
        scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer, milestones=[30, 40], gamma=0.5
        )

        epoch_scores = {"val": {}, "test": {}}

        for epoch in range(1, config.TRAIN.EPOCH + 1):
            loss, _ = train(train_loader, model, history, contrastive_module,
                            optimizer, device, data, config)

            val_auc, val_raw = test(valid_loader, model, history,
                                    contrastive_module, measure, device, data, config)
            test_auc, test_raw = test(test_loader, model, history,
                                      contrastive_module, measure, device, data, config)

            val_auc, test_auc = val_auc[0], test_auc[0]
            epoch_scores["val"][epoch] = {**val_raw, "auc": val_auc}
            epoch_scores["test"][epoch] = {**test_raw, "auc": test_auc}

            mlflow.log_metric("train_loss", loss, step=epoch)
            mlflow.log_metric("val_auc", val_auc, step=epoch)
            mlflow.log_metric("test_auc", test_auc, step=epoch)

            if val_auc > best_val:
                best_val, best = val_auc, test_auc

                if config.TEST.SAVE_WEIGHT:
                    torch.save(model.state_dict(), "best_model.pt")
                    torch.save(history.state_dict(), "best_history.pt")
                    mlflow.log_artifact("best_model.pt")
                    mlflow.log_artifact("best_history.pt")

                mlflow.log_metric("best_val_auc", best_val)
                mlflow.log_metric("best_test_auc", best)
                mlflow.log_metric("best_epoch", epoch)

            msg = (f"Epoch: {epoch:03d}, Loss: {loss:.4f}, "
                   f"Val AUC: {val_auc:.2%}, Test AUC: {test_auc:.2%}, "
                   f"Best AUC: {best:.2%}")
            print(msg)
            log.info(msg)

            scheduler.step()

        torch.save(epoch_scores, "log/raw_scores_history.pt")


def train(loader, model, history, contrastive_module, optimizer, device, data, config):
    model.train()
    history.train()
    contrastive_module.train()

    total_loss = 0.0
    log_detail = {"L1_temporal": 0.0, "L2_structural": 0.0, "L3_prototype": 0.0}

    pbar = tqdm(loader)
    for batch in pbar:
        batch = batch.to(device)
        optimizer.zero_grad()

        z = model(batch)

        root_global = data.src[batch.input_id.cpu()].to(device)
        src_local = root_local_indices(root_global, batch.n_id,
                                       data.num_nodes, device)
        h_src = z[src_local]

        cur_mem, prev_mem = history(h_src, root_global)
        if config.TRAIN.ERROR:
            cur_mem = cur_mem + 0.01 * torch.randn_like(cur_mem)

        # Temporal InfoNCE: pull the current memory towards its previous
        # state, push it away from random node histories.
        num_neg = batch.batch_size
        neg_mem = history.get_history(
            torch.randint(0, history.num_nodes, size=(num_neg,))
        )
        loss_temp = contrastive_loss(prev_mem, cur_mem, 0.02)
        loss_temp += torch.exp(
            cosine_similarity(cur_mem, neg_mem)
        ).sum(dim=1).log().mean()

        loss, detail = contrastive_module(
            h_current=cur_mem,
            h_graph=z,
            history_deque=history.history,
            node_idxs=root_global,
            sampled_node_ids=batch.n_id,
            edge_index=batch.edge_index,
            w1=0,
            w2=1.0,
            w3=0,
        )
        loss = loss + loss_temp

        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            list(model.parameters()) + list(history.parameters()), max_norm=1.0
        )
        optimizer.step()

        total_loss += loss.item()
        for k in log_detail:
            log_detail[k] += detail[k]

        pbar.set_description(
            f"Loss={loss.item():.4f} "
            f"L1={detail['L1_temporal']:.4f} "
            f"L2={detail['L2_structural']:.4f} "
            f"L3={detail['L3_prototype']:.4f}"
        )

    n = len(loader)
    return total_loss / n, {k: v / n for k, v in log_detail.items()}


@torch.no_grad()
def test(loader, model, history, contrastive_module, measure, device, data, config):
    model.eval()
    history.eval()
    contrastive_module.eval()

    preds, labels = [], []
    s_temporals, s_structurals = [], []

    for batch in tqdm(loader):
        batch = batch.to(device)

        z = model(batch)

        root_global = data.src[batch.input_id.cpu()].to(device)
        src_local = root_local_indices(root_global, batch.n_id,
                                       data.num_nodes, device)
        h_src = z[src_local]
        label = batch.y[: batch.batch_size]

        cur_mem, prev_mem = history(h_src, root_global, update=True)
        if config.TEST.ERROR:
            cur_mem = cur_mem + 0.01 * torch.randn_like(cur_mem)

        # Temporal score: deviation of the current memory from its past state.
        pred = torch.zeros(cur_mem.size(0), device=device)
        if config.TEST.ANOMALY.PRED:
            pred = 1 - torch.diag(cosine_similarity(cur_mem, prev_mem))

        if config.TEST.ANOMALY.FUNC:
            pred = pred + weighted_anomaly_score(history.history, root_global)

        # Structural score, restricted to the prediction nodes.
        s_structural = structural_anomaly_score(h=z, edge_index=batch.edge_index)
        s_structural = s_structural[src_local]

        score = (pred + s_structural).sigmoid()

        preds.append(score.cpu())
        labels.append(label.cpu())
        s_temporals.append(pred.cpu())
        s_structurals.append(s_structural.cpu())

    preds = torch.cat(preds)
    labels = torch.cat(labels)
    s_temporals = torch.cat(s_temporals)
    s_structurals = torch.cat(s_structurals)

    preds[torch.isnan(preds)] = 0.0

    # Nodes labelled 2 are unknown and excluded from evaluation.
    mask = labels != 2
    auc = measure(labels[mask], preds[mask])

    raw = {
        "s_temporal": s_temporals[mask],
        "s_structural": s_structurals[mask],
        "labels": labels[mask],
    }
    return auc, raw


if __name__ == "__main__":
    main()