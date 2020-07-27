import os.path as osp
import numpy as np
from easydict import EasyDict
from utils import project_root



cfg = EasyDict()
# TEST CONFIGS
cfg.MODEL = ''

#NUM_STEPS = 500 # Number of images in the validation set.
cfg.INFO_TARGET = str(project_root / 'bl/data/data_list/seqs_02_list/info.json')
cfg.RESTORE_FROM = None
cfg.visGPU = ''
cfg.SET = 'val'

cfg.TEST = EasyDict()

cfg.TEST.START_ITER = 0
cfg.TEST.MAX_ITERS = 0
cfg.TEST.STEP = 2000
cfg.TEST.MODEL_DIR = ''
cfg.TEST.WAIT_MODEL = True


# Test sets
cfg.TEST.BATCH_SIZE_TARGET = 1
cfg.TEST.INPUT_SIZE = (512, 256)
