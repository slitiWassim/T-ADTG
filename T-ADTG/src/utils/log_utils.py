import os
import time
import logging
from pathlib import Path
from typing import NamedTuple

import mlflow
import mlflow.pytorch


class RunPaths(NamedTuple):
    """Filesystem layout for a single training run."""
    run_name: str
    output_dir: Path      # output/{dataset}/{run_name}/
    checkpoints: Path
    metrics: Path
    log_file: Path         # log/{run_name}.log


def create_logger(config):
    """Creates the output/log folders for a run and a matching logger.

    Layout produced:
        {OUTPUT_DIR}/{dataset}/{dataset}_{timestamp}/
            checkpoints/     -> model + history weights
            metrics/         -> per-epoch metric dumps (.npz)
        {LOG_DIR}/{dataset}_{timestamp}.log

    Falls back to "output" / "log" if `config.OUTPUT_DIR` / `config.LOG_DIR`
    aren't defined.
    """
    dataset = config.DATASET.NAME
    time_str = time.strftime('%Y-%m-%d-%H-%M-%S')
    run_name = f'{dataset}_{time_str}'

    output_root = Path(getattr(config, 'OUTPUT_DIR', 'output'))
    log_root = Path(getattr(config, 'LOG_DIR', 'log'))

    run_dir = output_root / dataset / run_name
    checkpoints_dir = run_dir / 'checkpoints'
    metrics_dir = run_dir / 'metrics'
    for d in (run_dir, checkpoints_dir, metrics_dir):
        d.mkdir(parents=True, exist_ok=True)

    log_root.mkdir(parents=True, exist_ok=True)
    log_file = log_root / f'{run_name}.log'

    logger = logging.getLogger(run_name)
    logger.setLevel(logging.INFO)
    logger.handlers.clear()  # avoid duplicate handlers if called more than once
    logger.propagate = False

    fmt = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

    file_handler = logging.FileHandler(str(log_file))
    file_handler.setFormatter(fmt)
    logger.addHandler(file_handler)

    return logger, RunPaths(run_name=run_name, output_dir=run_dir,
                            checkpoints=checkpoints_dir, metrics=metrics_dir,
                            log_file=log_file)


class _NoOpRun:
    """Stand-in for the mlflow run context when tracking is disabled."""
    def __enter__(self):
        return self

    def __exit__(self, *exc_info):
        return False


class MLflowTracker:
    """Thin wrapper around mlflow that becomes a no-op when disabled, so
    the training loop never has to branch on whether tracking is on."""

    def __init__(self, enabled: bool):
        self.enabled = enabled

    def start_run(self, **kwargs):
        return mlflow.start_run(**kwargs) if self.enabled else _NoOpRun()

    def set_experiment(self, name):
        if self.enabled:
            mlflow.set_experiment(name)

    def set_tag(self, *args, **kwargs):
        if self.enabled:
            mlflow.set_tag(*args, **kwargs)

    def log_params(self, *args, **kwargs):
        if self.enabled:
            mlflow.log_params(*args, **kwargs)

    def log_metric(self, *args, **kwargs):
        if self.enabled:
            mlflow.log_metric(*args, **kwargs)

    def log_artifact(self, *args, **kwargs):
        if self.enabled:
            mlflow.log_artifact(*args, **kwargs)

    def log_model(self, *args, **kwargs):
        if self.enabled:
            mlflow.pytorch.log_model(*args, **kwargs)