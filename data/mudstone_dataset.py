import os
import sys
import torch
import numpy as np
from PIL import Image
from torchvision import transforms as T
from torch.utils.data import Dataset, DataLoader
from torchvision.utils import save_image
import os.path as osp
import cv2 as cv



class MUDSTONE_Dataset(Dataset):
    '''
    root : The folder contains all the download images
    '''

    def __init__(self, root, img_list_path, max_iters=None, mean=(128, 128, 128),
                 transform=None):
        '''
        '''
        self.dataset = "mudstone dataset"
        self.root = root
        self.img_list_path = img_list_path
        self.max_iters = max_iters
        self.transform = transform
        self.mean = mean
        self.img_ids = [i_id.strip() for i_id in open(img_list_path)]
        if not max_iters==None:
            self.img_ids = self.img_ids * int(np.ceil(float(max_iters) / len(self.img_ids)))
        self.files = []
        for img_id in self.img_ids:
            img_name, lbl = img_id.split(' ')
            img_file = osp.join(self.root, img_name)
            self.files.append({
                "img": img_file,
                "label": int(lbl),
                "name": img_name
            })


    def __len__(self):
        return len(self.img_ids)


    def __getitem__(self, index):
        datafiles = self.files[index]

        image = Image.open(datafiles['img']).convert('RGB')
        #image = cv.imread(datafiles['img']) #BGR

        #image = Image.fromarray(cv.cvtColor(image, cv.COLOR_BGR2RGB))
        #image = Image.open(datafiles['img']).convert('RGB') # open img and convert to "RGB"
        label = datafiles['label']
        name = datafiles["name"]

        # transform
        if self.transform != None:
            image = self.transform(image)

        image = image[:, :, ::-1]  # change to BGR
        image = image.transpose((2, 0, 1))

        return torch.from_numpy(image.copy()).float(), label, name



if __name__ == "__main__":
    from data.utils.transform import Compose, RandomSized_and_Crop


    transform = Compose([RandomSized_and_Crop([512, 256])])

    root = r'/opt/data-set/public/ycwang/'
    img_list_path = r'/mnt/groupproflizhen/wangyichao/petrographic_classification/data/data_list/train_list.txt'
    dataset = MUDSTONE_Dataset(root, img_list_path, max_iters=10000, transform=transform)
    dataloader = DataLoader(dataset,  batch_size=2, shuffle=True, drop_last=True)
    loader_iter = enumerate(dataloader)
    for i in range(5):
        idx_s, source_batch = next(loader_iter)
        img, label, name = source_batch
        print(img.size())
        print(label)
        if i > 2:
            break