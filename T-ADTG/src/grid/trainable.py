import numpy as np
import torch
import mlflow

from src.datasets import load_dataset, make_split_loaders
from src.models import TADTG
from src.utils.memory import Memory
from src.utils.contrastive import MultiLevelContrastiveLoss
from src.utils.metric import Metric
from src.utils.train_utils import set_seed, train, inference



# Apply hyperparameters to a YACS config
def apply_params(cfg, params):
    cfg.defrost()

    for key, value in params.items():
        keys = key.split(".")
        node = cfg

        for k in keys[:-1]:
            node = getattr(node, k)

        setattr(node, keys[-1], value)

    cfg.freeze()


def run_trial(cfg, device="cuda:0", trial=None, logger=None):

    set_seed(42)
    device = torch.device(device)


    # Load dataset
    data = load_dataset(cfg.DATASET.NAME, zero_edge=cfg.DATASET.ZERO_EDGE)
    train_loader, valid_loader, test_loader = make_split_loaders(data, cfg)

    metrics = Metric(["auroc", "auprc"])


    memory = Memory(
        data.num_nodes,
        config=cfg,
        device=device,
    ).to(device)

    model = TADTG(
        num_nodes=data.num_nodes,
        in_dim=data.x.size(1),
        edge_dim=data.msg.size(1),
        config=cfg,
    ).to(device)

    contrastive_module = MultiLevelContrastiveLoss(
        num_nodes=data.num_nodes,
        config=cfg,
        device=device,
    ).to(device)


    # Optimizer + scheduler
    optimizer = torch.optim.Adam(
        list(model.parameters())
        + list(memory.parameters())
        + list(contrastive_module.parameters()),
        lr=cfg.TRAIN.LR,
        weight_decay=cfg.TRAIN.WD,
    )

    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer,
        milestones=[10, 20],
        gamma=cfg.TRAIN.LOSS.GAMMA,
    )


    # Tracking best results
    best_val_auc = 0.0
    best_test_auc = 0.0
    best_epoch = 0

    for epoch in range(1, cfg.TRAIN.EPOCH + 1):

        loss = train(
            train_loader, model, memory, contrastive_module,
            optimizer, device, data, cfg,
        )

        val_dict = inference(
            valid_loader, model, memory, contrastive_module,
            metrics, device, data, cfg,
        )
        test_dict = inference(
            test_loader, model, memory, contrastive_module,
            metrics, device, data, cfg,
        )

        val_auc, val_auprc = val_dict["auroc"], val_dict["auprc"]
        test_auc, test_auprc = test_dict["auroc"], test_dict["auprc"]


        mlflow.log_metric("train_loss", loss, step=epoch)
        mlflow.log_metric("val_auroc", val_auc, step=epoch)
        mlflow.log_metric("test_auroc", test_auc, step=epoch)
        mlflow.log_metric("val_auprc", val_auprc, step=epoch)
        mlflow.log_metric("test_auprc", test_auprc, step=epoch)


        # Save best based on validation
        if val_auc > best_val_auc:
            best_val_auc = val_auc
            best_test_auc = test_auc
            best_epoch = epoch

            mlflow.log_metric("best_val_auroc", best_val_auc, step=epoch)
            mlflow.log_metric("best_test_auroc", best_test_auc, step=epoch)

            if logger:
                logger.info(
                    f"NEW BEST @ {epoch} | "
                    f"val={best_val_auc:.4f} test={best_test_auc:.4f}"
                )

            if cfg.TEST.SAVE_WEIGHT:
                metrics_path = f"metric_epoch_{epoch}.npz"
                np.savez(metrics_path, **test_dict)
                mlflow.log_artifact(metrics_path, artifact_path="roc_metrics")

        if logger:
            logger.info(
                f"Epoch {epoch} | "
                f"loss={loss:.4f} | "
                f"val={val_auc:.4f} | "
                f"test={test_auc:.4f}"
            )

        scheduler.step()

    if logger:
        logger.info(
            f"FINISHED | "
            f"best_val={best_val_auc:.4f} "
            f"best_test={best_test_auc:.4f} "
            f"epoch={best_epoch}"
        )

    return best_val_auc, best_test_auc, best_epoch