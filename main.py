import os
import cv2
import shutil
import random
import argparse
import numpy as np
from data_loader import get_loader
from solver import Solver

import torch
import torch.distributed as dist

def initiation():
    parser = argparse.ArgumentParser(description='Image Forgery Localization')
    parser.add_argument('--task', type=str, default='DataAugForensic')
    parser.add_argument('--amp', type=bool, default=False)
    parser.add_argument('--image_size', type=int, default=384)
    parser.add_argument('--test_size', type=int, default=640)
    parser.add_argument('--sliding_window_test',default=False)
    parser.add_argument('--swa', type=bool, default=True)

    parser.add_argument('--local_rank', type=int)
    parser.add_argument('--evaluation', action='store_true')
    parser.add_argument('--TTA', type=bool, default=True)
    parser.add_argument('--is_distributed', action='store_false')
    parser.add_argument('--model_path', type=str, default='./models/')
    parser.add_argument('--test_path', type=str, default='./sample_image/')
    parser.add_argument('--save_path', type=str, default='./save_path/')
    config = parser.parse_args()

    if not os.path.exists(config.save_path):
        os.makedirs(config.save_path)

    return config, None, None

def rm_and_make_dir(path):
    if os.path.exists(path):
        shutil.rmtree(path)
    os.makedirs(path)

def decompose(test_path, test_size):
    flist = os.listdir(test_path)
    size_list = [int(test_size)]
    path_out='temp/input_decompose_' + str(test_size) + '/'
    for size in size_list:
        path_out = 'temp/input_decompose_' + str(size) + '/'
        rm_and_make_dir(path_out)
    rtn_list = [[]]
    for file in flist:
        img = cv2.imread(test_path + file)
        H, W, _ = img.shape
        size_idx = 0
        while size_idx < len(size_list) - 1:
            if H < size_list[size_idx+1] or W < size_list[size_idx+1]:
                break
            size_idx += 1
        rtn_list[size_idx].append(file)
        size = size_list[size_idx]
        path_out = 'temp/input_decompose_' + str(size) + '/'
        X, Y = H // (size // 2) + 1, W // (size // 2) + 1
        idx = 0
        for x in range(X-1):
            if x * size // 2 + size > H:
                break
            for y in range(Y-1):
                if y * size // 2 + size > W:
                    break
                img_tmp = img[x * size // 2: x * size // 2 + size, y * size // 2: y * size // 2 + size, :]
                cv2.imwrite(path_out + file[:-4] + '_%03d' % idx + file[-4:], img_tmp)
                idx += 1
            img_tmp = img[x * size // 2: x * size // 2 + size, -size:, :]
            cv2.imwrite(path_out + file[:-4] + '_%03d' % idx + file[-4:], img_tmp)
            idx += 1
        for y in range(Y - 1):
            if y * size // 2 + size > W:
                break
            img_tmp = img[-size:, y * size // 2: y * size // 2 + size, :]
            cv2.imwrite(path_out + file[:-4] + '_%03d' % idx + file[-4:], img_tmp)
            idx += 1
        img_tmp = img[-size:, -size:, :]
        cv2.imwrite(path_out + file[:-4] + '_%03d' % idx + file[-4:], img_tmp)
        idx += 1
    return path_out

def main(config, work_init_fn, g):
    config.local_rank=int(os.environ['LOCAL_RANK'])
    if config.is_distributed:
        config.nprocs = torch.cuda.device_count()
        dist.init_process_group(backend='nccl',init_method='env://')
        device=torch.device('cuda:{}'.format(config.local_rank))
        torch.cuda.set_device(device)
    else:
        device=torch.device('cuda:{}'.format(config.local_rank))
        torch.cuda.set_device(device)

    loader=None
    va_loader=None
    if config.evaluation == False:
        pass
    else:
        if config.sliding_window_test:
            tmp_de_path=decompose(test_path=config.test_path, test_size=config.test_size)
            print('Decomposition complete.')
        else:
            tmp_de_path=config.test_path
        
        loader = get_loader(image_path=tmp_de_path,
                            mode='launch',
                            batch_size=1,
                            image_size=config.test_size,
                            num_workers=0,
                            work_init_fn=work_init_fn,
                            generator=g,
                            is_distributed=config.is_distributed)

    solver=Solver(config, loader, va_loader)

    solver.launch('./models/swa_model_trained_on_CASIAv2.tar', tmp_de_path)

if __name__ == '__main__':
    config,work_init_fn,g=initiation()
    main(config,work_init_fn,g)