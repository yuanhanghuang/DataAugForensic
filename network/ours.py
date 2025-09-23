import torch
import torch.nn as nn
from torch.nn import functional as F

import numpy as np
from torchvision import transforms as T
from torch import einsum
from einops import rearrange

from network.convnext import convnext_tiny,LayerNorm
from network.decoder import HamDecoder
from network.bricks import resize

from timm.models.layers import trunc_normal_, DropPath
    
def normalize_kernel2d(input: torch.Tensor) -> torch.Tensor:
    r"""Normalize both derivative and smoothing kernel."""
    if len(input.size()) < 2:
        raise TypeError(f"input should be at least 2D tensor. Got {input.size()}")
    norm: torch.Tensor = input.abs().sum(dim=-1).sum(dim=-1)
    return input / (norm.unsqueeze(-1).unsqueeze(-1))

def get_sobel(in_chan, out_chan, norm):
    filter_x = np.array([
        [1, 0, -1],
        [2, 0, -2],
        [1, 0, -1],
    ]).astype(np.float32)

    filter_y = np.array([
        [1, 2, 1],
        [0, 0, 0],
        [-1, -2, -1],
    ]).astype(np.float32)

    filter_x = filter_x.reshape((1, 1, 3, 3))
    filter_x = np.repeat(filter_x, in_chan, axis=1)
    filter_x = np.repeat(filter_x, out_chan, axis=0)

    filter_y = filter_y.reshape((1, 1, 3, 3))
    filter_y = np.repeat(filter_y, in_chan, axis=1)
    filter_y = np.repeat(filter_y, out_chan, axis=0)

    filter_x = torch.from_numpy(filter_x)
    filter_y = torch.from_numpy(filter_y)
    if norm == True:
        filter_x = normalize_kernel2d(filter_x)
        filter_y = normalize_kernel2d(filter_y)

    filter_x = nn.Parameter(filter_x, requires_grad=False)
    filter_y = nn.Parameter(filter_y, requires_grad=False)

    conv_x = nn.Conv2d(in_chan, out_chan, kernel_size=3, stride=1, padding=1, groups=1, bias=False)
    conv_x.weight = filter_x
    conv_y = nn.Conv2d(in_chan, out_chan, kernel_size=3, stride=1, padding=1, groups=1, bias=False)
    conv_y.weight = filter_y

    sobel_x = nn.Sequential(conv_x, LayerNorm(out_chan, eps=1e-6, data_format="channels_first"))
    sobel_y = nn.Sequential(conv_y, LayerNorm(out_chan, eps=1e-6, data_format="channels_first"))
    return sobel_x, sobel_y

class sobel_extra(nn.Module):
    def __init__(self, in_chan, out_chan, norm):
        super().__init__()

        self.sobel_x, self.sobel_y = get_sobel(in_chan=in_chan, out_chan=out_chan, norm=norm)

    def forward(self, input):
        g_x = self.sobel_x(input)
        g_y = self.sobel_y(input)
        g = torch.sqrt(torch.pow(g_x, 2) + torch.pow(g_y, 2) + 1e-6)

        return torch.sigmoid(g) * input

class FFAttention(nn.Module):
    def __init__(
        self,
        dim,
        heads,
        reduction_ratio,
        attn_drop=0., 
        proj_drop=0.,
    ):
        super().__init__()
        self.scale = (dim // heads) ** -0.5
        self.heads = heads
        self.reduction_ratio=reduction_ratio

        self.to_q = nn.Conv2d(dim, dim, 1, bias=False)
        self.to_kv = nn.Conv2d(dim, dim * 2, 1, bias=False)
        self.to_out = nn.Conv2d(dim, dim, 1, bias=False)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj_drop = nn.Dropout(proj_drop)

        if self.reduction_ratio > 1:
            self.sr = nn.Conv2d(dim, dim, kernel_size=reduction_ratio, stride=reduction_ratio, bias=False)
            self.norm = LayerNorm(dim, eps=1e-6, data_format="channels_first")

        self.gamma_s = nn.Parameter(torch.ones((1)))

    def forward(self, x, up=None):
        h, w = x.shape[-2:] 
        heads = self.heads 

        if up==None:
            up=x
        
        if self.reduction_ratio > 1:
            x= self.norm(self.sr(x))

        q, k, v = (self.to_q(up), *self.to_kv(x).chunk(2, dim = 1))
        q, k, v = map(lambda t: rearrange(t, 'b (h c) x y -> (b h) (x y) c', h = heads), (q, k, v))

        sim = einsum('b i d, b j d -> b i j', q, k) * self.scale
        attn = sim.softmax(dim = -1)
        attn=self.attn_drop(attn)

        out = einsum('b i j, b j d -> b i d', attn, v)
        out = rearrange(out, '(b h) (x y) c -> b (h c) x y', h = heads, x = h, y = w)
        return up + self.gamma_s * self.proj_drop(self.to_out(out))

class FFBlock(nn.Module):
    def __init__(self, dim, drop_path=0., layer_scale_init_value=1e-6):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim, bias=False) # depthwise conv
        self.norm = LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, 4 * dim, bias=False) # pointwise/1x1 convs, implemented with linear layers
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(4 * dim, dim, bias=False)
        self.gamma = nn.Parameter(layer_scale_init_value * torch.ones((dim)), 
                                    requires_grad=True) if layer_scale_init_value > 0 else None
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        input = x
        x = self.dwconv(x)
        x = x.permute(0, 2, 3, 1).contiguous()  
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        if self.gamma is not None:
            x = self.gamma * x
        x = x.permute(0, 3, 1, 2).contiguous()  

        x = input + self.drop_path(x)
        return x

class compress_up_block(nn.Module):
    def __init__(
        self,
        cur_chan,
        tar_chan,
        norm_or_not=False
    ):
        super().__init__()
        self.conv=nn.Conv2d(cur_chan, tar_chan, kernel_size=1,stride=1,padding=0, bias=False)
        self.act=nn.ReLU(inplace=True)


    def forward(self, x, tar_size=None):
        x=self.conv(x)
        if tar_size!=None:
            x=resize(x,size=tar_size)
        x=self.act(x)
        return x

class Fusion_block(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels
    ):
        super().__init__()
        self.conv = nn.Sequential(
                                nn.ReLU(inplace=True),  
                                nn.Conv2d(in_channels, out_channels, 
                                          kernel_size=3, padding=1, stride=1, bias=False),
                                LayerNorm(out_channels, eps=1e-6, data_format="channels_first")               
                                )

    def forward(self, p,i,d):
        edge_att = torch.sigmoid(d)
        return self.conv(edge_att*p + (1-edge_att)*i)


class NestedUNet(nn.Module):
    def __init__(self):
        super(NestedUNet, self).__init__()
        self.enc_embed_dims=[96, 192, 384, 768]
        self.decoder = HamDecoder(enc_embed_dims=self.enc_embed_dims)
        self.norm_layer0=LayerNorm(96, eps=1e-6, data_format="channels_first")
        self.norm_layer1=LayerNorm(192, eps=1e-6, data_format="channels_first")
        self.norm_layer2=LayerNorm(384, eps=1e-6, data_format="channels_first")
        self.norm_layer3=LayerNorm(768, eps=1e-6, data_format="channels_first")
   
        self.p_layer2=nn.Sequential(
            FFAttention(dim=192,heads=2,reduction_ratio=4),
            FFBlock(dim=192),
            FFAttention(dim=192,heads=2,reduction_ratio=4),
            FFBlock(dim=192),
            LayerNorm(192, eps=1e-6, data_format="channels_first")
        )
        self.attn_block2=FFAttention(dim=384,heads=4,reduction_ratio=2)
        self.compress_up_block2=compress_up_block(384,192)

        self.p_layer3=nn.Sequential(
            nn.Conv2d(192, 384, kernel_size=1, bias=False),
            FFAttention(dim=384,heads=4,reduction_ratio=4),#1
            FFBlock(dim=384),
            FFAttention(dim=384,heads=4,reduction_ratio=4),
            FFBlock(dim=384),
            LayerNorm(384, eps=1e-6, data_format="channels_first"),
        )
        self.attn_block3=FFAttention(dim=768,heads=8,reduction_ratio=1)
        self.compress_up_block3=compress_up_block(768,384)

        self.compress_hier_feat = nn.Sequential(
            nn.Conv2d(
                in_channels=1344,
                out_channels=1344,#//8
                kernel_size=1,
                stride=1,
                padding=0, bias=False),
            nn.GELU(),
            nn.Conv2d(
                in_channels=1344,
                out_channels=384,#config.DATASET.NUM_CLASSES,
                kernel_size=1,#extra.FINAL_CONV_KERNEL,
                stride=1,
                padding=0, bias=False),#if extra.FINAL_CONV_KERNEL == 3 else 0
            LayerNorm(384, eps=1e-6, data_format="channels_first")
        )

        self.sobel1=sobel_extra(96,96,True)
        self.d_layer1=nn.Sequential(
            FFBlock(dim=96),
            nn.Conv2d(96, 192, kernel_size=1, bias=False),
            LayerNorm(192, eps=1e-6, data_format="channels_first"),
        )

        self.sobel2=sobel_extra(192,192,True)
        self.d_layer2=nn.Sequential(
            FFBlock(dim=192),
            nn.Conv2d(192, 384, kernel_size=1, bias=False),
            LayerNorm(384, eps=1e-6, data_format="channels_first")
        )

        self.sobel3=sobel_extra(384,384,True)
        self.d_layer3=nn.Sequential(
            FFBlock(dim=384),
            LayerNorm(384, eps=1e-6, data_format="channels_first")
        )

        self.fusion_block=Fusion_block(384,384)

        self.cls_conv = nn.Sequential(
            nn.Conv2d(sum(self.enc_embed_dims[1:])//8, 1, kernel_size=1, bias=False))#//8 nn.Dropout2d(p=0.2),

        self.i_net = convnext_tiny()

    def forward(self, input):
        outputs=self.i_net(input)
        d_out=self.norm_layer0(outputs[0])
        outputs=outputs[1:]
 
        outputs[0]=self.norm_layer1(outputs[0])
        outputs[1]=self.norm_layer2(outputs[1])
        outputs[2]=self.norm_layer3(outputs[2])

        p_out=self.p_layer2[0](outputs[0],self.compress_up_block2(self.attn_block2(outputs[1]), outputs[0].shape[2:]))
        p_out=self.p_layer2[1:](p_out)
        p_out=self.p_layer3[0](p_out)
        p_out=self.p_layer3[1](p_out,self.compress_up_block3(self.attn_block3(outputs[2]), p_out.shape[2:]))
        p_out=self.p_layer3[2:](p_out)

        d_out=resize(d_out,size=(d_out.shape[2]//2,d_out.shape[3]//2),mode='bilinear')
        d_out=self.d_layer1(self.sobel1(d_out))+self.sobel2(outputs[0])
        d_out=self.d_layer2(d_out)+self.sobel3(resize(outputs[1],size=(d_out.shape[2:]),mode='bilinear'))
        d_out=self.d_layer3(d_out)

        outputs = [resize(output, size=(48,48), mode='bilinear') for output in outputs]#(48,48)
        outputs = torch.cat(outputs, dim=1)
        outputs = self.compress_hier_feat(outputs)

        p_out = resize(p_out, size=(48,48), mode='bilinear')
        d_out = resize(d_out,size=(48,48), mode='bilinear')
        outputs = self.fusion_block(p_out,outputs,d_out)

        # segmentation head
        output=self.decoder(outputs)
        output=self.cls_conv(output)
        output=resize(output,size=input.shape[2:],mode='bilinear')
        
        # projection head

        return output
