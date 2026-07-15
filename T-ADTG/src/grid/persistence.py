import yaml
import pandas as pd


def save_best_yaml(result):
    with open("best_grid_config.yaml", "w") as f:
        yaml.safe_dump(result, f)


def save_csv(results):
    df = pd.DataFrame(results)
    df.to_csv("grid_results.csv", index=False)