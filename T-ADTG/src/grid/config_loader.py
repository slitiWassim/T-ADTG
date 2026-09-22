import copy
from src.config.defaults import _C


def load_cfg_from_yaml(yaml_path: str):
    """
    Load config with priority:
    defaults.py -> yaml file
    """
    cfg = copy.deepcopy(_C)

    cfg.defrost()
    cfg.merge_from_file(yaml_path)
    cfg.freeze()

    return cfg