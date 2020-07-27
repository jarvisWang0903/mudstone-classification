import argparse
import os
import os.path as osp
import pprint
import random
import warnings
import numpy as np
import yaml
import torch
from utils.util import cfg_from_file
from options.train_options import cfg
from data import CreateDataLoader
from model import get_resnet, get_vgg
from classification.train_BL import train_baseline


warnings.filterwarnings("ignore", message="numpy.dtype size changed")
warnings.filterwarnings("ignore")


def get_arguments():
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser(description="Code for training")
    parser.add_argument('--yml', type=str, default='./scripts/train.yml',
                        help='optional config file', )
    return parser.parse_args()


def main():

    args = get_arguments()
    print('Called with args:')
    print(args)


    cfg_from_file(cfg, args.yml)

    if cfg.EXP_NAME == '':
        cfg.EXP_NAME = f'exp_1'


    if cfg.EXP_ROOT_SNAPSHOT == '':
        cfg.EXP_ROOT_SNAPSHOT = osp.join(cfg.EXP_ROOT, 'snapshots')
    if cfg.EXP_ROOT_LOGS == '':
        cfg.EXP_ROOT_LOGS = osp.join(cfg.EXP_ROOT, 'logs')
    if not osp.exists(cfg.EXP_ROOT_SNAPSHOT):
        os.makedirs(cfg.EXP_ROOT_SNAPSHOT)
    if not osp.exists(cfg.EXP_ROOT_LOGS):
        os.makedirs(cfg.EXP_ROOT_LOGS)

    print('Using config:')
    pprint.pprint(cfg)

    os.environ["CUDA_VISIBLE_DEVICES"] = cfg.visGPU


    # INIT
    _init_fn = None
    if not cfg.RANDOM_TRAIN:
        torch.manual_seed(cfg.TRAIN.RANDOM_SEED)
        torch.cuda.manual_seed(cfg.TRAIN.RANDOM_SEED)
        np.random.seed(cfg.TRAIN.RANDOM_SEED)
        random.seed(cfg.TRAIN.RANDOM_SEED)

        def _init_fn(worker_id):
            np.random.seed(cfg.TRAIN.RANDOM_SEED + worker_id)


    model_dict = {}

    # LOAD SEGMENTATION NET
    assert osp.exists(cfg.INIT_FROM), f'Missing init model {cfg.INIT_FROM}'
    if cfg.TRAIN.MODEL == 'RESNET':
        if cfg.METHOD == 'baseline':
            model_dict['model'] = get_resnet()
        else:
            raise NotImplementedError
    elif cfg.TRAIN.MODEL == 'VGG':
        if cfg.METHOD == 'baseline':
            model_dict['model'] = get_vgg()
        else:
            raise NotImplementedError
    else:
        raise NotImplementedError(f"Not yet supported {cfg.TRAIN.MODEL}")
    print('Model loaded')

    if cfg.RESTOR_FROM != '':
        state_dict = torch.load(cfg.RESTOR_FROM, map_location=lambda storage, loc: storage)
        if cfg.METHOD == 'baseline':
            model_dict['model'].load_state_dict(state_dict['model_state_dict'])
            print(f'model restore from {cfg.RESTOR_FROM}')
        else:
            raise NotImplementedError

    print('preparing dataloaders ...')
    dataloader = CreateDataLoader(cfg)
    dataloader_iter = enumerate(dataloader)

    with open(osp.join(cfg.EXP_ROOT_LOGS, 'train_cfg.yml'), 'w') as yaml_file:
        yaml.dump(cfg, yaml_file, default_flow_style=False)

    # UDA TRAINING
    if cfg.METHOD == 'baseline':
        train_baseline(model_dict, dataloader_iter, cfg)


if __name__ == '__main__':
    main()
