# Copyright (c) Carl. All rights reserved.
from ipaddress import v6_int_to_packed
from xmlrpc.client import Boolean, boolean
from matplotlib.ft2font import BOLD

from mmcls.models.backbones.earlyexit import mask_down
from torch import Tensor, zeros, ones, cuda
from torch import load, save, sum, max 
import torch.nn as nn
from torchvision.transforms import Grayscale

import pdb

from ..builder import BACKBONES
from . import ResNet

@BACKBONES.register_module()
class GRGBnet_Base(nn.Module):
    
    """ The BaseBlock of my BranchyNet Version. 
        It contains the first layer and the first exit.
        It also keeps hold of the whole loaded ResNet.

    """

    def __init__(self, use_grayscale: Boolean = True, use_rgb: Boolean = True):

        super(GRGBnet_Base, self).__init__()

        assert(use_grayscale or use_rgb)
        
        self.use_grayscale = use_grayscale
        self.use_rgb = use_rgb
        
        if cuda.is_available():
            self.device = 'cuda'
        else: 
            self.device = 'cpu'

        if self.use_rgb:
            self.model_rgb       = ResNet(depth=18, in_channels=3)
        if self.use_grayscale:
            self.model_grayscale = nn.Sequential(
                ResNet(depth=18, in_channels=1),
            )    
            self.grayscale = Grayscale()

        self.grayscale_head = nn.Sequential(
            nn.Conv2d(512, 512, 3, 2, padding=1),
            nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.ReLU(),
            # nn.Conv2d(512, 512, 3, 1, padding=0),
            # nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            # nn.ReLU(),
            nn.AvgPool2d(2, stride=1, padding=0),
            nn.Flatten(),
            nn.Linear(4608, 10),
            nn.Softmax(dim = 1))

        self.rgb_head = nn.Sequential(
            nn.AvgPool2d(3, stride=2, padding=1),
            nn.Flatten(),
            nn.Linear(4*4*512, 10),
            nn.Softmax(dim = 1))


    def forward(self, img: Tensor, return_loss: Boolean) -> Tensor:
        img = img.to(self.device)

        if return_loss:
            return self.forward_train(img)
        else:
            return self.forward_test(img)

    def forward_train(self, x: Tensor) -> Tensor:
        """ The standard use case for train is with rgb and grayscale. But 
            other settings are possible."""
        pdb.set_trace()
        if self.use_rgb:  
            y1 = self.model_rgb(x)[0]
            y1 = self.rgb_head(y1)
            if self.use_grayscale:
                y2 = self.model_grayscale(self.grayscale(x))[0]
                y2 = self.grayscale_head(y2)
                
                return [y1, y2]
            
            return y1    
        
        y2 = self.model_grayscale(self.grayscale(x))[0]
        y2 = self.grayscale_head(y2)

        return y2

    def simple_test(self, x: Tensor) -> Tensor:
        return forward_test(self, x)
    
    def forward_test(self, x: Tensor)-> Tensor:
        bs = x.size()[0]
        mask_grayscale = ones(bs).to(self.device)
        y = zeros(bs, 10).to(self.device)

        if self.use_grayscale:
            x_gray = self.grayscale(x)
            x_gray = self.model_grayscale(x_gray)[0]
            grayscale_output = self.grayscale_head(x_gray)
            
            if not self.use_rgb:
                return grayscale_output
            else:
                mask_grayscale = (max(grayscale_output, axis=1)[0] >= 0.80)
                y += mask_grayscale.reshape(-1,1) * grayscale_output 

        if self.use_rgb:        
            mask_grayscale = mask_grayscale <= 0.5
            x = mask_down(x, mask_grayscale)
            pdb.set_trace()
            x = self.model_rgb(x)[0]
            rgb_output = mask_up(self.rgb_head(x), mask_grayscale) 

            y += rgb_output
        
        return y

def mask_down(t: Tensor, mask: Tensor) -> Tensor:
    pdb.set_trace()
    mask = mask.reshape(-1)

    return t[mask.bool()]

def mask_up(t: Tensor, mask: Tensor) -> Tensor:
    '''This method takes a downsized vector and upsizes it again, so that the new tensor
        has its values where the mask has its Ones.'''
    pdb.set_trace()
    mask = mask.reshape(-1)
    
    bs = mask.size()[0]
    output = zeros(bs, *(list(t.size())[1: ]))

    output[mask] = t

    return output

    
    
            
