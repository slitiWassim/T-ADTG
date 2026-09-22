import itertools


GRID_SPACE = {
    "TRAIN.BATCH_SIZE_PER_GPU" : [64,128,256,512],
    "MODEL.NUM_LAYERS": [1,2,3],
    "DATASET.NUM_NEIGHBORRS": [[10,5,5],[15,10,5],[20,10,5],[25,15,5],[30,15,5]],
    "TRAIN.LR": [5e-4,1e-4,1e-3,5e-3],
    "MODEL.DROPOUT": [0.2,0.3,0.5],
    "MODEL.HEADS" : [4,8,16],
    "MODEL.HIDDEN_CHANNELS": [16,32,64,128,256],

}


def generate_combinations():
    keys = list(GRID_SPACE.keys())
    values = list(GRID_SPACE.values())

    combos = []
    for vals in itertools.product(*values):
        combos.append(dict(zip(keys, vals)))

    return combos