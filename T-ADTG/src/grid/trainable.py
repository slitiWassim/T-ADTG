import random
import numpy as np
import torch
import mlflow
import mlflow.pytorch

# import your existing functions from main.py
#from src.main_v1 import train, test 
from src.main_v4 import train,  test
from src.datasets.dataset import load_dataset
from src.datasets.loader import train_val_test_load
from src.utils.history import History
from src.utils.measure import Measure
from src.utils.contrastive import MultiLevelContrastiveLoss
from src.models.model_5 import EnhancedTemporalGNN


# =====================================================
# Reproducibility
# =====================================================
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


# =====================================================
# Apply hyperparameters to YACS config
# =====================================================
def apply_params(cfg, params):

    cfg.defrost()

    for key, value in params.items():

        keys = key.split(".")
        node = cfg

        for k in keys[:-1]:
            node = getattr(node, k)

        setattr(node, keys[-1], value)

    cfg.freeze()


# =====================================================
# Main training wrapper for one grid experiment
# =====================================================
def run_trial(cfg, device="cuda:0", trial=None, logger=None):

    set_seed(42)

    device = torch.device(device)

    # ======================================
    # Load dataset
    # ======================================
    data = load_dataset(
        cfg.DATASET.NAME,
        zero_edge=cfg.DATASET.ZERO_EDGE
    )

    train_loader, valid_loader, test_loader = train_val_test_load(
        data, cfg
    )

    measure = Measure("auc")

    # ======================================
    # Build History
    # ======================================
    history = History(
        data.num_nodes,
        cfg.DATASET.NUM_TIMESLOTS,
        cfg.MODEL.HIDDEN_CHANNELS,
        device=device,
        history_retrieve=cfg.HISTORY.RETRIEVE,
        recurrent=cfg.HISTORY.RECURRENT,
        normalize=cfg.HISTORY.NORMALIZE,
    ).to(device)

    # ======================================
    # Build Model
    # ======================================
    model = EnhancedTemporalGNN(
        num_nodes=data.num_nodes,
        in_dim=data.x.size(1),
        edge_dim=data.msg.size(1),
        hidden_dim=cfg.MODEL.HIDDEN_CHANNELS,
        num_layers=cfg.MODEL.NUM_LAYERS,
        heads=cfg.MODEL.HEADS,
        dropout=cfg.MODEL.DROPOUT,
        window=cfg.MODEL.WINDOW,
        memory_momentum=cfg.TRAIN.MOMENTUM,
        t2v_dim=32
    ).to(device)

    # ======================================
    # Contrastive Loss Module
    # ======================================
    contrastive_module = MultiLevelContrastiveLoss(
        num_nodes=data.num_nodes,
        dimension=cfg.MODEL.HIDDEN_CHANNELS,
        temperature=cfg.TRAIN.TEMPRATURE,
        ema_decay=cfg.TRAIN.EM_DECAY,
        num_negatives=cfg.TRAIN.LOSS.NUM_NEGATIVES,
        device=device,
    ).to(device)

    # ======================================
    # Optimizer
    # ======================================
    optimizer = torch.optim.Adam(
        list(model.parameters())
        + list(history.parameters())
        + list(contrastive_module.parameters()),
        lr=cfg.TRAIN.LR,
        weight_decay=cfg.TRAIN.WD
    )

    # ======================================
    # Tracking Best Results
    # ======================================
    best_val = 0.0
    best_test = 0.0
    best_epoch = 0
    best = 0

    # ======================================
    # Epoch Loop
    # ======================================
    
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer, 
            milestones=[ 10 , 20 ], 
            gamma=cfg.TRAIN.LOSS.GAMMA)
    
    for epoch in range(1, cfg.TRAIN.EPOCH + 1):

        loss, detail = train(
            train_loader,
            model,
            history,
            contrastive_module,
            optimizer,
            device,
            data,
            cfg
        )

        val_auc , _ = test(
            valid_loader,
            model,
            history,
            contrastive_module,
            measure,
            device,
            data,
            cfg
        )

        test_auc , metrics = test(
            test_loader,
            model,
            history,
            contrastive_module,
            measure,
            device,
            data,
            cfg
        )

        # -----------------------------
        # Log metrics per epoch
        # -----------------------------
        
        mlflow.log_metric("train_loss", loss, step=epoch)
        mlflow.log_metric("val_auc", val_auc, step=epoch)
        mlflow.log_metric("test_auc", test_auc, step=epoch)

        # detailed losses
        for k, v in detail.items():
            mlflow.log_metric(k, v, step=epoch)

        # -----------------------------
        # Save best based on validation
        # -----------------------------
        if test_auc > best :

            mlflow.log_metric("best_auc", test_auc)
            mlflow.log_metric("best_auc_epoch", epoch)
            best_epoch =  epoch

            if logger:
                logger.info(
                    f"BEST AUC In The XP @ {epoch} | "
                    f"test={test_auc:.4f}")   

        if val_auc > best_val:
            best_val = val_auc
            best = test_auc
                
            if cfg.TEST.SAVE_WEIGHT :
                """
                torch.save(model.state_dict(), "best_model.pt")
                mlflow.log_artifact("best_model.pt")


                torch.save(history , "model_history.pt")
                mlflow.log_artifact("model_history.pt")

                mlflow.pytorch.log_model(
                        pytorch_model=model,
                        artifact_path="best_model")   
                    
                mlflow.log_metric("best_val_auc", best_val)
                mlflow.log_metric("best_test_auc", best)
                mlflow.log_metric("epoch", epoch)
                    
                mlflow.pytorch.log_model(
                    pytorch_model=history,
                    artifact_path="model_history") 
                    
                mlflow.log_metric("best_val_auc", best_val)
                mlflow.log_metric("best_test_auc", best)
                mlflow.log_metric("epoch", epoch)
                """
                    
                file_name = f"metric_epoch_{epoch}.npz"
                np.savez(file_name, **metrics)
                # Log file to MLflow
                mlflow.log_artifact(file_name, artifact_path="roc_metrics")



            if logger:
                logger.info(
                    f"NEW BEST @ {epoch} | "
                    f"val={best_val:.4f} "
                    f"test={best:.4f}")    

            mlflow.log_metric("best_val_auc", best_val)
            mlflow.log_metric("best_test_auc", best)
            mlflow.log_metric('best_epoch',epoch)
        
        
        if logger:
            logger.info(
                f"Epoch {epoch} | "
                f"loss={loss:.4f} | "
                f"val={val_auc:.4f} | "
                f"test={test_auc:.4f}")

        scheduler.step()
    # ======================================
    # Final log
    # ======================================
    #mlflow.log_metric("best_val_auc", best_val)
    #mlflow.log_metric("best_test_auc", best)
    #mlflow.log_metric("best_epoch", best_epoch)
    
    if logger:
        logger.info(
            f"FINISHED | "
            f"best_val={best_val:.4f} "
            f"best_test={best:.4f} "
            f"epoch={best_epoch}" )

    return best_val, best, best_epoch