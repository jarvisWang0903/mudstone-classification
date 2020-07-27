import os
import sys
from pathlib import Path
import os.path as osp
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
import torch.optim as optim
from tensorboardX import SummaryWriter
from torch import nn
from tqdm import tqdm

from utils.func import adjust_learning_rate
from utils.loss import loss_calc

from utils.util import print_losses, log_losses_tensorboard

def train_baseline(model_dict, trainloader_iter, cfg):


    device = cfg.GPU_ID

    viz_tensorboard = os.path.exists(cfg.EXP_ROOT_LOGS)
    if viz_tensorboard:
        writer = SummaryWriter(log_dir=cfg.EXP_ROOT_LOGS)

    model = model_dict['model']

    #  NETWORK
    model.train()
    model.to(device)

    cudnn.benchmark = True
    cudnn.enabled = True


    # OPTIMIZERS
    model_optim = optim.SGD(model.optim_parameters(cfg.TRAIN.LEARNING_RATE),
                                     lr=cfg.TRAIN.LEARNING_RATE,
                                     momentum=cfg.TRAIN.MOMENTUM,
                                     weight_decay=cfg.TRAIN.WEIGHT_DECAY)



    for i_iter in tqdm(range(cfg.TRAIN.EARLY_STOP + 1)):

        # reset optimizers
        model_optim.zero_grad()

        # adapt LR if needed
        adjust_learning_rate(model_optim, i_iter, cfg)

        # get data
        _, batch = trainloader_iter.__next__()
        images, labels, name = batch

       ##training code


        model_optim.step()

        current_losses = {'seg_s': loss}

        print_losses(current_losses, cfg, i_iter)


        # 存储模型
        if i_iter % cfg.TRAIN.SAVE_PRED_EVERY == 0 and i_iter != 0:
            print('taking snapshot ...')
            print('exp =', cfg.EXP_ROOT_SNAPSHOT)
            snapshot_dir = Path(cfg.EXP_ROOT_SNAPSHOT)
            torch.save({
                'step': i_iter + 1,
                'model_state_dict': model.state_dict(),
            }, snapshot_dir / f'model_{i_iter}.pth.tar')
            if i_iter >= cfg.TRAIN.EARLY_STOP - 1:
                break
        sys.stdout.flush()

        # Visualize with tensorboard
        if viz_tensorboard:
            log_losses_tensorboard(writer, current_losses, i_iter)







