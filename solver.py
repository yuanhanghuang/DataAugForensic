import os
import random
import copy
import cv2
from utils import *
import numpy as np
np.seterr(divide='ignore',invalid='ignore')
import shutil
from tqdm import tqdm
from visualize import Visualizer
from network.ours import NestedUNet

from tensorboardX import SummaryWriter
import torch
from torch.cuda.amp import autocast
from torch.nn import functional as F
from torch.nn.parallel import DistributedDataParallel
import torch.distributed as dist
from torchvision import transforms as T
torch.autograd.set_detect_anomaly(False)

class Solver(object):
    def __init__(self, config, loader, va_loader):
        self.loader = loader
        self.va_loader=va_loader
        self.config=config
        
        if self.config.evaluation==False:
            if self.config.is_distributed:
                if dist.get_rank()==0:
                    self.writer = SummaryWriter(comment='{}'.format(self.config.task))
                    self.vis = Visualizer(self.config.task)
                
        self.build_model()

    def build_model(self):
        if self.config.is_distributed:
            self.unet = NestedUNet().cuda()
            self.unet = DistributedDataParallel(self.unet, find_unused_parameters=True)
        else:
            self.unet = NestedUNet().cuda()

        if self.config.is_distributed:
            if self.config.local_rank==0:
                self.print_network(self.unet)
        else:
            self.print_network(self.unet)

    def print_network(self, model):
        """Print out the network information."""
        num_params = 0
        for p in model.parameters():
            num_params += p.numel()
        print("The number of parameters: {}".format(num_params))

    def image_visdom(self,image):
        image[:,0,:,:] = (image[:,0,:,:]*0.269+0.413)*255.0
        image[:,1,:,:] = (image[:,1,:,:]*0.262+0.406)*255.0
        image[:,2,:,:] = (image[:,2,:,:]*0.283+0.369)*255.0
        return image.type(torch.uint8)

    def GT_visdom(self,GT):
    	GT = (GT*255.).type(torch.uint8)
    	return GT

    def SR_visdom(self,SR):
    	SR = (SR*255.).type(torch.uint8)
    	return SR

    def launch(self, path, tmp_de_path):
        checkpoint = torch.load(path, map_location=torch.device('cpu'))
        if self.config.is_distributed:
            if self.config.swa and 'swa' in path:
                checkpoint['model'] = {key.replace("module.module.", "module."): value for key, value in checkpoint['model'].items()}
                for k in list(checkpoint['model'].keys()):
                    if k.startswith('n_averaged'):
                        print('n_averaged:',checkpoint['model'][k])
                        del checkpoint['model'][k]
                    elif k.startswith('module.'):
                        continue
                    else:
                        print('error in loading swa model checkpoint')
            self.unet.load_state_dict(checkpoint['model'])
        else:
            if self.config.swa and 'swa' in path:
                checkpoint['model'] = {key.replace("module.module.", ""): value for key, value in checkpoint['model'].items()}
                for k in list(checkpoint['model'].keys()):
                    if k.startswith('n_averaged'):
                        print('n_averaged:',checkpoint['model'][k])
                        del checkpoint['model'][k]
            else:
                checkpoint['model'] = {key.replace("module.", ""): value for key, value in checkpoint['model'].items()}
            self.unet.load_state_dict(checkpoint['model'],strict=True)

        if self.config.TTA == True:
            import ttach as tta
            self.unet = tta.SegmentationTTAWrapper(self.unet, tta.aliases.d4_transform(), merge_mode='mean')

        self.unet.eval()
        tmp_pred_path = 'temp/input_decompose_' + str(self.config.test_size) + '_pred/'
        rm_and_make_dir(tmp_pred_path)
        with torch.no_grad():
            for i, (image, GT, filename) in tqdm(enumerate(self.loader), total=len(self.loader)):
                if self.config.sliding_window_test==False:
                    if image.shape[2] > self.config.test_size or image.shape[3] > self.config.test_size:   
                        image = F.interpolate(image, size=(self.config.image_size, self.config.image_size), mode='bilinear', align_corners=True)
                    else:
                        pass
                
                image = image.cuda(non_blocking=True)
                with autocast(enabled=self.config.amp):
                    SR = self.unet(image)

                SR = torch.sigmoid(SR.float())
                image_path = tmp_pred_path+filename[0] if self.config.sliding_window_test \
                    else self.config.save_path + filename[0] 
                cv2.imwrite(image_path,
                		np.squeeze((SR*255.).permute(0, 2, 3, 1).cpu().detach().numpy().astype(np.uint8)))
            print('Prediction complete.')
            if os.path.exists(tmp_de_path) and self.config.sliding_window_test==True:
                shutil.rmtree(tmp_de_path)
                path_pre = merge(self.config.test_path, save_path=self.config.save_path+self.config.test_path.split('/')[-2], test_size=self.config.test_size)
                print('Merging complete.')
            if os.path.exists(tmp_pred_path):
                shutil.rmtree(tmp_pred_path)
    

def gkern(kernlen=7, nsig=3):
    """Returns a 2D Gaussian kernel."""
    # x = np.linspace(-nsig, nsig, kernlen+1)
    # kern1d = np.diff(st.norm.cdf(x))
    # kern2d = np.outer(kern1d, kern1d)
    # rtn = kern2d/kern2d.sum()
    # rtn = np.concatenate([rtn[..., None], rtn[..., None], rtn[..., None]], axis=2)
    rtn = [[0, 0, 0],
           [0, 1, 0],
           [0, 0, 0]]
    rtn = np.array(rtn, dtype=np.float32)
    # rtn = np.concatenate([rtn[..., None], rtn[..., None], rtn[..., None]], axis=2)
    rtn = cv2.resize(rtn, (kernlen, kernlen))
    return rtn

def rm_and_make_dir(path):
    if os.path.exists(path):
        shutil.rmtree(path)
    os.makedirs(path)


def merge(test_path, save_path, test_size):
    path_d = 'temp/input_decompose_' + str(test_size) + '_pred/'
    path_r = save_path
    size = int(test_size)

    gk = gkern(size)
    gk = 1 - gk

    for file in os.listdir(test_path):
        img = cv2.imread(test_path + file)
        H, W, _ = img.shape
        X, Y = H // (size // 2) + 1, W // (size // 2) + 1
        idx = 0
        rtn = np.ones((H, W), dtype=np.float32) * -1  #, 1
        for x in range(X-1):
            if x * size // 2 + size > H:
                break
            for y in range(Y-1):
                if y * size // 2 + size > W:
                    break
                img_tmp = cv2.imread(path_d + file[:-4] + '_%03d' % idx + file[-4:], cv2.IMREAD_UNCHANGED)
                # img_tmp = cv2.cvtColor(img_tmp, cv2.COLOR_GRAY2BGR)
                # print(img_tmp.shape)
                weight_cur = copy.deepcopy(rtn[x * size // 2: x * size // 2 + size, y * size // 2: y * size // 2 + size])
                h1, w1 = weight_cur.shape
                gk_tmp = cv2.resize(gk, (w1, h1))
                weight_cur[weight_cur != -1] = gk_tmp[weight_cur != -1]
                weight_cur[weight_cur == -1] = 0
                weight_tmp = copy.deepcopy(weight_cur)
                weight_tmp = 1 - weight_tmp
                rtn[x * size // 2: x * size // 2 + size, y * size // 2: y * size // 2 + size] = weight_cur * rtn[x * size // 2: x * size // 2 + size, y * size // 2: y * size // 2 + size] + weight_tmp * img_tmp
                idx += 1
            img_tmp = cv2.imread(path_d + file[:-4] + '_%03d' % idx + file[-4:], cv2.IMREAD_UNCHANGED)
            # img_tmp = cv2.cvtColor(img_tmp, cv2.COLOR_GRAY2BGR)
            weight_cur = copy.deepcopy(rtn[x * size // 2: x * size // 2 + size, -size:])
            h1, w1 = weight_cur.shape
            gk_tmp = cv2.resize(gk, (w1, h1))
            weight_cur[weight_cur != -1] = gk_tmp[weight_cur != -1]
            weight_cur[weight_cur == -1] = 0
            weight_tmp = copy.deepcopy(weight_cur)
            weight_tmp = 1 - weight_tmp
            rtn[x * size // 2: x * size // 2 + size, -size:] = weight_cur * rtn[x * size // 2: x * size // 2 + size, -size:] + weight_tmp * img_tmp
            idx += 1
        for y in range(Y - 1):
            if y * size // 2 + size > W:
                break
            img_tmp = cv2.imread(path_d + file[:-4] + '_%03d' % idx + file[-4:], cv2.IMREAD_UNCHANGED)
            # img_tmp = cv2.cvtColor(img_tmp, cv2.COLOR_GRAY2BGR)
            weight_cur = copy.deepcopy(rtn[-size:, y * size // 2: y * size // 2 + size])
            h1, w1 = weight_cur.shape
            gk_tmp = cv2.resize(gk, (w1, h1))
            weight_cur[weight_cur != -1] = gk_tmp[weight_cur != -1]
            weight_cur[weight_cur == -1] = 0
            weight_tmp = copy.deepcopy(weight_cur)
            weight_tmp = 1 - weight_tmp
            rtn[-size:, y * size // 2: y * size // 2 + size] = weight_cur * rtn[-size:, y * size // 2: y * size // 2 + size] + weight_tmp * img_tmp
            idx += 1
        img_tmp = cv2.imread(path_d + file[:-4] + '_%03d' % idx + file[-4:], cv2.IMREAD_UNCHANGED)
        # img_tmp = cv2.cvtColor(img_tmp, cv2.COLOR_GRAY2BGR)
        weight_cur = copy.deepcopy(rtn[-size:, -size:])#, :
        h1, w1 = weight_cur.shape#, _
        gk_tmp = cv2.resize(gk, (w1, h1))
        weight_cur[weight_cur != -1] = gk_tmp[weight_cur != -1]
        weight_cur[weight_cur == -1] = 0
        weight_tmp = copy.deepcopy(weight_cur)
        weight_tmp = 1 - weight_tmp
        rtn[-size:, -size:] = weight_cur * rtn[-size:, -size:] + weight_tmp * img_tmp
        idx += 1
        cv2.imwrite(path_r +'/'+ file[:-4]+'.png', rtn)
    return path_r