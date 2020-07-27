import numpy as np
from torch.utils import data
from data.utils.transform import Compose, RandomSized_and_Crop
from data.mudstone_dataset import MUDSTONE_Dataset

__all__=['CreateDataLoader']



def CreateDataLoader(cfg, set='train'):

    input_transform = Compose([RandomSized_and_Crop(cfg.TRAIN.INPUT_SIZE)])

    if cfg.SET=='train':#stone_list/train_list_shuffle.txt
        dataset = MUDSTONE_Dataset(cfg.ROOT, img_list_path='./data_list/stone_list/train_list_shuffle.txt',
                                   max_iters=cfg.TRAIN.MAX_ITERS * cfg.TRAIN.BATCH_SIZE, transform=input_transform)
        dataloader = data.DataLoader(dataset, batch_size=cfg.TRAIN.BATCH_SIZE, shuffle=True,
                                            num_workers=cfg.NUM_WORKERS, pin_memory=True)
    elif set=='val':
        dataset = MUDSTONE_Dataset(cfg.ROOT, img_list_path='./data_list/stone_list/val_list.txt',
                                   max_iters=None, transform=None)
        dataloader = data.DataLoader(dataset, batch_size=1, shuffle=False,
                                     num_workers=cfg.NUM_WORKERS, pin_memory=True)
    else:
        raise NotImplementedError

    return dataloader

if __name__ == '__main__':
    from utils.util import cfg_from_file
    from options.train_options import cfg
    import argparse

    def get_arguments():
        """
        Parse input arguments
        """
        parser = argparse.ArgumentParser(description="Code for training")
        parser.add_argument('--yml', type=str, default='../scripts/train.yml',
                            help='optional config file', )
        return parser.parse_args()


    args = get_arguments()
    print('Called with args:')
    print(args)

    cfg_from_file(cfg, args.yml)
    print('preparing dataloaders ...')
    dataloader = CreateDataLoader(cfg)
    dataloader_iter = enumerate(dataloader)
    _, batch = dataloader_iter.__next__()
    images, labels, name = batch
    print(images.size())
    print(labels.size())
    print(name)