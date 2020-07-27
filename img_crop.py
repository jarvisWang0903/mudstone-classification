import os
import sys
import numpy as np
from PIL import Image
import os.path as osp
import cv2 as cv

root = 'E:\\'
img_list_path = r'C:\Users\Jarvis\Desktop\petrographic classification\data\data_list\stone_list\val_list.txt'
save_path = r'E:\img_crop'

img_ids = [i_id.strip() for i_id in open(img_list_path)]
files = []
print(len(img_ids))
for img_id in img_ids:
    img_name, lbl = img_id.split(' ')
    img_file = osp.join(root, img_name)
    image = cv.imread(img_file)  # BGR
    array = cv.cvtColor(image, cv.COLOR_BGR2RGB) # HxWxC 3200,4096,3
    if array.shape == (3200, 4096, 3):
        new_img = array[0:3000, :, :]
        img_list = []
        for i in range(1000, 3001, 1000):
            for j in range(1365, 4096, 1365):
                _img = array[i-1000:i, j-1365:j, :]
                img_list.append(_img)
        print(len(img_list))
    elif array.shape == (3000, 4096, 3):
        img_list = []
        for i in range(1000, 3001, 1000):
            for j in range(1365, 4096, 1365):
                _img = array[i - 1000:i, j - 1365:j, :]
                img_list.append(_img)
        print(len(img_list))
    else:
        raise NotImplementedError

    assert len(img_list)==9

    for index in range(len(img_list)):
        img_save = Image.fromarray(img_list[index])
        img_save.save('%s\%s.png' % (save_path, img_name.rsplit('.')[0]+ f"_{index}"))
