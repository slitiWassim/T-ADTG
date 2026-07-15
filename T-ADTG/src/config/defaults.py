from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from yacs.config import CfgNode as CN


_C = CN()

_C.DATA_DIR = 'data'
_C.LOG_DIR = 'log'
_C.GPUS = (0, 1, 2, 3)
_C.WORKERS = 4



# DATASET related params
_C.DATASET = CN()
_C.DATASET.ROOT = '../../data'
_C.DATASET.NAME = 'wikipedia'
_C.DATASET.PROCESSED = 'processed'
_C.DATASET.RAW = 'raw'
_C.DATASET.NUM_NEIGHBORRS = [15,10]
_C.DATASET.NUM_TIMESLOTS = 14
_C.DATASET.ZERO_EDGE = False

# train
_C.TRAIN = CN()

_C.TRAIN.BATCH_SIZE_PER_GPU = 256
_C.TRAIN.SHUFFLE = True

_C.TRAIN.EPOCH = 10
_C.TRAIN.CHECKPOINT = ''

_C.TRAIN.OPTIMIZER = 'adam'

# sgd and
_C.TRAIN.MOMENTUM = 0.9
_C.TRAIN.WD = 1e-5
_C.TRAIN.NESTEROV = False
_C.TRAIN.TEMPRATURE = 0.07
_C.TRAIN.EM_DECAY = 0.99

_C.TRAIN.LR_TYPE = 'linear'
_C.TRAIN.LR = 0.0002
_C.TRAIN.LR_STEP = [40, 70]
_C.TRAIN.LR_FACTOR = 0.5
_C.TRAIN.ERROR =  0.01
_C.TRAIN.RATIO =  0.7


## Contrastive Loss Function
_C.TRAIN.LOSS = CN()
_C.TRAIN.LOSS.TAU = 0.02
_C.TRAIN.LOSS.ALPHA = 1
_C.TRAIN.LOSS.BETA = 1
_C.TRAIN.LOSS.GAMMA = 1.0
_C.TRAIN.LOSS.NUM_NEGATIVES = 64
_C.TRAIN.LOSS.W1 = 1
_C.TRAIN.LOSS.W2 = 0.5
_C.TRAIN.LOSS.W3 = 0.5






_C.HISTORY = CN()
_C.HISTORY.RETRIEVE = 'last' 
_C.HISTORY.RECURRENT = 'gru'
_C.HISTORY.NORMALIZE = True
_C.HISTORY.OUT_NORMALIZE = False

## test
_C.TEST = CN()

_C.TEST.BATCH_SIZE_PER_GPU = 256
_C.TEST.SHUFFLE = False 
_C.TEST.ERROR = False
_C.TEST.SAVE_WEIGHT = False
_C.TEST.ANOMALY = CN()
_C.TEST.ANOMALY.PRED = 1
_C.TEST.ANOMALY.FUNC = 0.0
_C.TEST.ANOMALY.SIGMOID = True
_C.TEST.ANOMALY.WDECAY = 1.0


# common params for NETWORK
_C.MODEL = CN()
_C.MODEL.NAME = 'TAG-AD'
_C.MODEL.INIT_WEIGHTS = True
_C.MODEL.PRETRAINED = ''
_C.MODEL.NUM_LAYERS = 2
_C.MODEL.HIDDEN_CHANNELS = 64
_C.MODEL.HEADS = 8
_C.MODEL.DROPOUT = 0.3
_C.MODEL.WINDOW = 8
# _C.MODEL.SIGMA = 1.5



def update_config(cfg, args):
    cfg.defrost()
    cfg.merge_from_file(args.cfg)
    cfg.merge_from_list(args.opts)
    cfg.freeze()


### TO DO : optimize to return parameters in dict format all in the args dict istead of all beeing in the 'config' variable 

def flatten_cfg(cfg, parent_key='', sep='.'):
    items = {}

    if isinstance(cfg, CN):
        cfg = cfg.items()
    elif isinstance(cfg, dict):
        cfg = cfg.items()
    else:
        return {parent_key: cfg} if parent_key else cfg

    for k, v in cfg:
        new_key = f"{parent_key}{sep}{k}" if parent_key else k

        if isinstance(v, (CN, dict)):
            items.update(flatten_cfg(v, new_key, sep=sep))
        else:
            items[new_key] = v

    return items 


if __name__ == '__main__':
    import sys
    with open(sys.argv[1], 'w') as f:
        print(_C, file=f)
