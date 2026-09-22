import argparse

import numpy as np
import torch

from src.config.defaults import _C as config, update_config, flatten_cfg
from src.datasets import load_dataset, make_split_loaders
from src.models import TADTG
from src.utils.memory import Memory
from src.utils.contrastive import MultiLevelContrastiveLoss
from src.utils.metric import Metric
from src.utils.log_utils import create_logger, MLflowTracker
from src.utils.train_utils import set_seed, train, inference 



def parse_args():
    parser = argparse.ArgumentParser(description='T-ADTG')
    parser.add_argument('--cfg', help='experiment configuration filename',
                        default='src/config/wikipedia.yaml', type=str)
    parser.add_argument('--gpu', type=int, default=0, help='GPU id')
    parser.add_argument('--exp', type=str, default=None,
                        help='Run name for MLflow. If omitted, MLflow logging is disabled.')
    parser.add_argument('opts',
                        help="Modify config options using the command-line",
                        default=None,
                        nargs=argparse.REMAINDER)

    args = parser.parse_args()
    update_config(config, args)
    return args


def main():
    set_seed(42)  # Encourage reproducible experiments
    args = parse_args()

    device = torch.device(f"cuda:{args.gpu}")
    print(f"Using {device}")

    ## Define logger
    log, paths = create_logger(config)
    log.info(f'----------------> Start Experiment <----------------')
    log.info(f'Dataset: {config.DATASET.NAME} | Device: {device} |  Output: {paths.output_dir}  |  Log file: {paths.log_file}')

    ## Load datasets
    data = load_dataset(config.DATASET.NAME, zero_edge=config.DATASET.ZERO_EDGE)
    train_loader, valid_loader, test_loader = make_split_loaders(data, config)

    ## Define evaluation metrics
    metrics = Metric(["auroc", "auprc"])

    ## Log experiment using mlflow
    mlf = MLflowTracker(enabled=args.exp is not None)
    mlf.set_experiment(config.DATASET.NAME)

    with mlf.start_run(run_name=args.exp or config.MODEL.NAME):
        mlf.log_params(flatten_cfg(config))
        mlf.set_tag("model", config.MODEL.NAME)
        mlf.set_tag("experiment_description",
                    "A Time-Aware Self-Supervised Framework for Anomaly Detection in Temporal Graphs")

        memory = Memory(
            data.num_nodes,
            config = config,
            device=device).to(device)

        model = TADTG(
            num_nodes=data.num_nodes,
            in_dim=data.x.size(1),
            edge_dim=data.msg.size(1),
            config=config).to(device)
        arch_file = paths.output_dir / "model_architecture.txt"
        arch_file.write_text(str(model))
        mlf.log_artifact(str(arch_file))


        contrastive_module = MultiLevelContrastiveLoss(
            num_nodes=data.num_nodes,
            config=config,
            device=device).to(device)

        optimizer = torch.optim.Adam(
            list(model.parameters())
            + list(memory.parameters())
            + list(contrastive_module.parameters()),
            lr=config.TRAIN.LR,
            # A small weight decay helps prevent trivial collapse in InfoNCE
            weight_decay=config.TRAIN.WD,
        )

        best_val_auc, best_test_auc = 0.0, 0.0

        for epoch in range(1, config.TRAIN.EPOCH + 1):
            loss = train(train_loader, model, memory, contrastive_module,
                                 optimizer, device, data, config)
            ## Validation
            val_dict = inference(valid_loader, model, memory, contrastive_module,
                            metrics, device, data, config)
            ## Test
            test_dict = inference(test_loader, model, memory, contrastive_module,
                                 metrics, device, data, config)

            val_auc, val_auprc = val_dict["auroc"], val_dict["auprc"]
            test_auc, test_auprc = test_dict["auroc"], test_dict["auprc"]

            mlf.log_metric("train_loss", loss, step=epoch)
            mlf.log_metric("val_auroc", val_auc, step=epoch)
            mlf.log_metric("test_auroc", test_auc, step=epoch)
            mlf.log_metric("val_auprc", val_auprc, step=epoch)
            mlf.log_metric("test_auprc", test_auprc, step=epoch)

            if val_auc > best_val_auc:
                best_val_auc = val_auc
                best_test_auc = test_auc
                best_test_auprc = test_auprc
                mlf.log_metric("best_val_auroc", best_val_auc, step=epoch)
                mlf.log_metric("best_test_auroc", best_test_auc, step=epoch)

                # AUPRC recorded at the same epoch the AUROC improved
                mlf.log_metric("best_val_auprc", val_auprc, step=epoch)
                mlf.log_metric("best_test_auprc", test_auprc, step=epoch)

                if config.TEST.SAVE_WEIGHT:
                    model_path = paths.checkpoints / "best_model.pt"

                    torch.save(model.state_dict(), model_path)
                    mlf.log_artifact(str(model_path))
                    mlf.log_model(model, artifact_path="best_model")

                    metrics_path = paths.metrics / f"metric_epoch_{epoch}.npz"
                    np.savez(metrics_path, **test_dict)
                    mlf.log_artifact(str(metrics_path), artifact_path="metrics")

            msg = (
                f"Epoch: {epoch:03d}, Loss:{loss:.4f}, "
                f"Val AUROC:{val_auc:.2%}, Val AUPRC:{val_auprc:.2%}, "
                f"Test AUROC:{test_auc:.2%}, Test AUPRC:{test_auprc:.2%}, "
                f"Best AUROC:{best_test_auc:.2%}, Best AUPRC:{best_test_auprc:.2%}"
            )
            print(msg)
            log.info(msg)

            ## Reset memory and similarity 
            model.reset_memory()
            memory.init_memory()


if __name__ == '__main__':
    main()