import torch
import torch.nn as nn
import torch.nn.functional as F
from options.test_options import cfg
import json
import os.path as osp
import os
import numpy as np
from tqdm import tqdm
from utils.serialization import pickle_dump, pickle_load
import time
import argparse
from model import get_resnet,get_vgg
from utils.util import cfg_from_file
from data import CreateDataLoader

def get_arguments():
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser(description="Code for domain adaptation (DA) test")
    parser.add_argument('--yml', type=str, default='./scripts/test.yml',
                        help='optional config file', )
    return parser.parse_args()



def main():

    args = get_arguments()
    print('Called with args:')
    print(args)

    cfg_from_file(cfg, args.yml)

    if cfg.TEST.SAVE_PRED:
        save_path = cfg.TEST.SAVE
        if not os.path.exists(save_path):
            os.makedirs(save_path)

    os.environ["CUDA_VISIBLE_DEVICES"] = cfg.visGPU

    if cfg.MODEL == 'RESNET':
        encoder_common = get_resnet()
    elif cfg.MODEL == 'VGG':
        encoder_common = get_vgg()
    else:
        raise NotImplementedError

    valloader = CreateDataLoader(cfg)


    ##把结果存到pkl中
    cache_path = osp.join(cfg.TEST.MODEL_DIR, 'all_res.pkl')
    if osp.exists(cache_path):
        all_res = pickle_load(cache_path)
    else:
        all_res = {}
    cur_best = -1
    cur_best_model = ''
    # 从start_iter,到max_iter计算miou
    for i_iter in range(cfg.TEST.START_ITER, cfg.TEST.MAX_ITERS + 1, cfg.TEST.STEP):
        model_restore_from = osp.join(cfg.TEST.MODEL_DIR, f'model_{i_iter}.pth.tar')
        if not osp.exists(model_restore_from):
            # continue
            if cfg.TEST.WAIT_MODEL:
                print('Waiting for model..!')
                while not osp.exists(model_restore_from):
                    time.sleep(5)
                time.sleep(5)
        print("Evaluating model", model_restore_from)
        if i_iter not in all_res.keys():
            if model_restore_from is not None:
                state_dict = torch.load(model_restore_from, map_location=lambda storage, loc: storage)
                model.load_state_dict(state_dict['model_state_dict'])
                model.eval()
                model.cuda()
            
            #[cfg.TEST.EPISODE-1]
            for index, batch in tqdm(enumerate(targetloader)):
                image, label, _, name, _ = batch
                n, c, h, w = image.shape
                image = image.cuda()
                with torch.no_grad():
                    _, t_pred = encoder_common(image)
                    output = F.softmax(t_pred, dim=1)
                    output = F.interpolate(output, (h, w), mode='bilinear', align_corners=True).cpu().data[0].numpy()
                    output = output.transpose(1, 2, 0)
                    output_nomask = np.asarray(np.argmax(output, axis=2), dtype=np.uint8)
                assert output_nomask.shape == label[0].shape
                label = np.array(label[0], dtype=np.int64)
                if len(label.flatten()) != len(output_nomask.flatten()):
                    print('Skipping: len(gt) = {:d}, len(pred) = {:d}, {:s}'.format(len(label.flatten()),
                                                                                          len(output_nomask.flatten()),
                                                                                          name))
                    continue
                hist += fast_hist(label.flatten(), output_nomask.flatten(), num_classes)  #https://blog.csdn.net/buyaobianlan2017/article/details/80513628
                if index > 0 and index % 100 == 0:
                    with open(model_restore_from+'_mIoU.txt', 'a') as f:
                        f.write('{:d} / {:d}: {:0.2f}\n'.format(index, 1000, 100*np.mean(per_class_iu(hist))))
                    print('{:d} / {:d}: {:0.2f}'.format(index, 1000, 100*np.mean(per_class_iu(hist))))
                # 保存pred结果
                if cfg.TEST.SAVE_PRED:
                    name = name[0].split('/')[-1]
                    with open('%s/%s' % (save_path, 'pred_list.txt'), 'a') as fin:
                        fin.write(str(name) + '\n')
                    fin.close()
                    image_save = img_inv_preprocess(image, img_mean)
                    label = colorize_mask(label)
                    output_col = colorize_mask(output_nomask)
                    image_save.save('%s/%s_cimage.png' % (save_path, name.split('.')[0]))
                    label.save('%s/%s_blabel.png' % (save_path, name.split('.')[0]))
                    output_col.save('%s/%s_acolor.png' % (save_path, name.split('.')[0]))
                #计算mious并存到all_res.pkl中
                mIoUs = per_class_iu(hist)
                all_res[i_iter] = mIoUs
                pickle_dump(all_res, cache_path)
        else:
            mIoUs = all_res[i_iter]
        #计算每一类的mIoU
        for ind_class in range(num_classes):
            with open(model_restore_from + '_mIoU.txt', 'a') as f:
                f.write('===>' + name_classes[ind_class] + ':\t' + str(round(mIoUs[ind_class] * 100, 2)) + '\n')
            print('===>' + name_classes[ind_class] + ':\t' + str(round(mIoUs[ind_class] * 100, 2)))
        mIoU = round(np.nanmean(mIoUs) * 100, 2)

        with open(osp.join(cfg.TEST.MODEL_DIR, 'all_mIoU.txt'), 'a') as f:
            step = model_restore_from.rsplit('/', 1)[1]
            f.write('step: '+ str(step) + '\n')
            f.write('===> mIoU13: ' + str(mIoU) + '\n')

        print('===> mIoU13: ' + str(mIoU))
        if cur_best_miou < mIoU:
            cur_best_miou = mIoU
            cur_best_model = model_restore_from
        print('\tCurrent mIoU:', mIoU)
        print('\tCurrent best model:', cur_best_model)
        print('\tCurrent best mIoU:', cur_best_miou)


if __name__ == '__main__':
    main()


