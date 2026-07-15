import queue as q
import torch
import mlflow

from src.grid.config_loader import load_cfg_from_yaml
from src.grid.trainable import run_trial, apply_params , set_seed
from src.grid.utils import setup_trial_logger
from src.config.defaults import flatten_cfg


def gpu_worker(gpu_id, task_queue, yaml_path):
    """
    One persistent worker pinned to one GPU.
    Pulls experiments from queue until empty.
    """

    torch.cuda.set_device(gpu_id)
    torch.cuda.init()
    
    set_seed(42)
    
    while True:

        try:
            exp_id, params = task_queue.get_nowait()

        except q.Empty:
            break

        logger, log_path = setup_trial_logger(
            f"trial_{exp_id}"
        )

        cfg = load_cfg_from_yaml(yaml_path)
        apply_params(cfg, params)

        try:
            with mlflow.start_run(
                nested=True,
                run_name=f"grid_{exp_id}"
            ):

                mlflow.log_param("gpu_id", gpu_id)
                mlflow.log_param("yaml_config", yaml_path)

                #mlflow.log_params(params)
                mlflow.log_params(flatten_cfg(cfg)) 

                best_val, best_test, best_epoch = run_trial(
                    cfg,
                    device=f"cuda:{gpu_id}",
                    logger=logger
                )

                mlflow.log_metric(
                    "best_val_auc", best_val
                )
                mlflow.log_metric(
                    "best_test_auc", best_test
                )
                mlflow.log_metric(
                    "best_epoch", best_epoch
                )

                # attach log file
                mlflow.log_artifact(log_path)

                logger.info(
                    f"FINISHED | "
                    f"VAL={best_val:.4f} | "
                    f"TEST={best_test:.4f} | "
                    f"EPOCH={best_epoch}"
                )

        except Exception as e:
            logger.exception(
                f"Trial {exp_id} failed"
            )