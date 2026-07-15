import argparse
import torch.multiprocessing as mp
import mlflow

from src.grid.search_space_wikipedia import generate_combinations
from src.grid.worker import gpu_worker
from src.grid.trainable import set_seed


# --------------------------------------
# Hardware config
# --------------------------------------
GPUS = [0,1]

# workers per GPU
PER_GPU = 3


def parse_args():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--cfg",
        type=str,
        required=True,
        help="Dataset yaml config file"
    )

    parser.add_argument(
        "--experiment",
        type=str,
        default="GRID_MOOC",
        help="MLflow experiment name"
    )

    return parser.parse_args()


def main():
    
    set_seed(42)
    args = parse_args()

    combos = generate_combinations()

    print(f"Total combinations: {len(combos)}")

    manager = mp.Manager()
    queue = manager.Queue()

    for i, params in enumerate(combos):
        queue.put((i, params))

    mlflow.set_experiment(args.experiment)

    processes = []

    for gpu_id in GPUS:

        for _ in range(PER_GPU):

            p = mp.Process(
                target=gpu_worker,
                args=(gpu_id, queue, args.cfg)
            )

            p.start()
            processes.append(p)

    for p in processes:
        p.join()

    print("Grid Search Finished")


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    main()