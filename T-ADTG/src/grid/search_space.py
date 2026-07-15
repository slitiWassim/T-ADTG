import itertools


GRID_SPACE = {
#    "TRAIN.BATCH_SIZE_PER_GPU" : [256,512],
    "MODEL.NUM_LAYERS": [2,3],
    "DATASET.NUM_NEIGHBORRS": [[15,10],[30,15],[20,10],[30,10,5]],
    "TRAIN.LR": [5e-4,1e-4,1e-3,5e-3],
    "MODEL.DROPOUT": [0.2,0.3,0.5,0.7],
    "DATASET.NUM_TIMESLOTS" :[8,10,12,14,18,20,24],
    "MODEL.HEADS" : [4,8,16],
    "MODEL.HIDDEN_CHANNELS": [64,128,256],
    "TRAIN.ERROR": [0.001,0.0001,0.0],
#    "TRAIN.LOSS.GAMMA" : [0.75,0.5]
}


def generate_combinations():
    keys = list(GRID_SPACE.keys())
    values = list(GRID_SPACE.values())

    combos = []
    for vals in itertools.product(*values):
        combos.append(dict(zip(keys, vals)))

    return combos