import logging
import os


def setup_trial_logger(name: str):
    """
    One logger file per trial
    """
    os.makedirs("logs", exist_ok=True)

    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    file_path = f"logs/{name}.log"

    fh = logging.FileHandler(file_path)
    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(message)s"
    )

    fh.setFormatter(formatter)
    logger.addHandler(fh)

    logger.propagate = False

    return logger, file_path