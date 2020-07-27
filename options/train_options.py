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
cfg.EXP_ROOT_SNAPSHOT = ''    #模型存放文件夹
cfg.EXP_ROOT_LOGS = ''        # log文件夹# CUDA
cfg.GPU_ID = 0
cfg.visGPU = ''

cfg.METHOD = ''              #实验方法 baseline or ...
cfg.INIT_FROM = ''           # image net fine-tune地址
cfg.RESTOR_FROM = ''         # 断开恢复训练
cfg.SET = 'train'           #train or val
cfg.ROOT = r''              # 图片存放地址


# TRAIN CONFIGS
cfg.TRAIN = EasyDict()
cfg.TRAIN.BATCH_SIZE = 1
cfg.TRAIN.INPUT_SIZE = [512, 256]
cfg.TRAIN.MAX_ITERS  = 250000
cfg.TRAIN.EARLY_STOP  = 150000

# Segmentation network params
cfg.TRAIN.MODEL = 'RESNET'          # 使用的模型 RESNET or VGG
cfg.TRAIN.LEARNING_RATE = 2.5e-4
cfg.TRAIN.MOMENTUM = 0.9
cfg.TRAIN.WEIGHT_DECAY = 0.0005
cfg.TRAIN.POWER = 0.9
cfg.TRAIN.LAMBDA_MAIN = 1.0         # loss 权重