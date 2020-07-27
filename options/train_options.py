import os.path as osp
import numpy as np
from easydict import EasyDict
from utils import project_root



cfg = EasyDict()

# COMMON CONFIGS

# Number of workers for dataloading
cfg.NUM_WORKERS = 4
cfg.RANDOM_TRAIN = False
# Exp dirs
cfg.EXP_NAME = ''
cfg.EXP_ROOT = ''
cfg.EXP_ROOT_SNAPSHOT = ''
cfg.EXP_ROOT_LOGS = ''
# CUDA
cfg.GPU_ID = 0
cfg.visGPU = ''

cfg.METHOD = ''
cfg.INIT_FROM = ''
cfg.RESTOR_FROM = ''
cfg.SET = 'train'
cfg.ROOT = r''


# TRAIN CONFIGS
cfg.TRAIN = EasyDict()
cfg.TRAIN.BATCH_SIZE = 1
cfg.TRAIN.INPUT_SIZE = [512, 256]
cfg.TRAIN.MAX_ITERS  = 250000
cfg.TRAIN.EARLY_STOP  = 150000

# Segmentation network params
cfg.TRAIN.MODEL = 'RESNET'
cfg.TRAIN.LEARNING_RATE = 2.5e-4
cfg.TRAIN.MOMENTUM = 0.9
cfg.TRAIN.WEIGHT_DECAY = 0.0005
cfg.TRAIN.POWER = 0.9
cfg.TRAIN.LAMBDA_MAIN = 1.0