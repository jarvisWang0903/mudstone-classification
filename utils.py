import os
import numpy as np
# file_path = r'img\Light_grey_fine_sandstone'
#
#
#
# path_ = r'E:\img\Light_grey_fine_sandstone'
#
#
# list_path = r'C:\Users\Jarvis\Desktop\petrographic classification\data\data_list\Light_grey_fine_sandstone_list.txt'
# file_fin = open(list_path,'w')


list_path = r'C:\Users\Jarvis\Desktop\petrographic classification\data\data_list\stone_list\train_list_new.txt'
list_path_s = r'C:\Users\Jarvis\Desktop\petrographic classification\data\data_list\stone_list\train_list_shuffle.txt'
import os.path as osp

#path_ = r'E:\img\Light_grey_fine_sandstone'

if __name__ == '__main__':
    # for root, dirs, files in os.walk(path_):
    #     for file in files:
    #         _name = str(file.split('-')[2] + '.' + file.split('-')[3].split('.')[-1])
    #         src = osp.join(path_,file)
    #         trg = osp.join(path_,_name)
    #         #name = os.path.join(file_path, _name)
    #         #file_fin.write(name + '\n')
    #         os.rename(src, trg)

    file_fin = open(list_path, 'r')
    file_fin_ = open(list_path_s, 'w')
    img_ids = [i_id for i_id in open(list_path)]
    np.random.seed(1)
    np.random.shuffle(img_ids)
    print(len(img_ids))
    for img_id in img_ids:
        file_fin_.write(img_id)
        # name, lbl = img_id.split(" ")
        # for i in range(9):
        #     new_name = name.rsplit(".")[0]+f'_{i}.png'
        #     file_fin.write(new_name + ' ' + lbl + '\n')
        # print(img_id)
