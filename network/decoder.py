#%%

from turtle import forward
import torch
import torch.nn.functional as F
import torch.nn as nn

from network.hamburger import HamBurger
from network.bricks import ConvBNReLU, resize
from network.convnext import LayerNorm
from torch.cuda.amp import autocast


class HamDecoder(nn.Module):
    '''SegNext'''
    def __init__(self, enc_embed_dims):
        super().__init__()

        # ham_channels = enc_embed_dims[-1]

        self.squeeze = ConvBNReLU(384, 336, 3)#1344  336 
        self.ham_attn = HamBurger(336)#336
        self.align = ConvBNReLU(336, 168, 3)#336 168
       
    def forward(self, x):
        
        # features = features[1:] # drop stage 1 features b/c low level
        # features = [resize(feature, size=features[-3].shape[2:], mode='bilinear') for feature in features]
        # x = torch.cat(features, dim=1)

        x = self.squeeze(x)
        # with autocast(enabled=False):
        #     x = x.float()
        x = self.ham_attn(x)
        x = self.align(x)
        # with autocast(enabled=True):
        #     x=x.half()
        
        # print(x.dtype)
    
        # print(x.dtype)
        # x=x.half()       
        return x


#%%

# import torch.nn.functional as F

# def resize(input,
#            size=None,
#            scale_factor=None,
#            mode='nearest',
#            align_corners=None,
#            warning=True):

#     return F.interpolate(input, size, scale_factor, mode, align_corners)

# inputs = [resize(
#         level,
#         size=x[0].shape[2:],
#         mode='bilinear',
#         align_corners=False
#     ) for level in x]

# for i in range(4):
#     print(x[i].shape)
# for i in range(4):
#     print(inputs[i].shape)



# inputs = torch.cat(inputs, dim=1)
# print(inputs.shape)
