
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as cp
from mmcv.cnn import build_conv_layer, build_norm_layer

from ..builder import BACKBONES, build_head, build_neck
from .resnet import Bottleneck, ResNet 
from .resnet import ResLayer, ResNetV1d
from ..heads.cls_head import ClsHead


@BACKBONES.register_module()
class Cascading():
    """Cascading backbone.

    """
    

    def __init__(self,
                pretrained_little=None,
                pretrained_big=None,
                alpha = 1.2,
                beta = 1,
                init_cfg = None,
                 **kwargs):
        head_dict = dict(
        type='LinearClsHead',
        num_classes=10,
        in_channels=512,
        loss=dict(type='CrossEntropyLoss', loss_weight=1.0),
        )
   
        self.little = ResNet(depth=18)
        #this should be looked how it works 
        self.little_head = ClsHead()
        self.pretrained_little = pretrained_little
        #set head here
        self.big = ResNet(depth=50)
        self.little_head =  build_head(head_dict)
        self.pretrained_big = pretrained_big
        self.big_head = build_head(head_dict)
        self.alpha = alpha
        self.beta = beta
        #set head here 
   
    def forward(self, x):
        y = self.little(x)
        score = y.topk(2)
        if (score.values[0][0] - score.values[0][1]) < self.threshold:
            y = self.big(x)
        return y


       
    def train(self, mode=True):
        if self.pretrained_little:
            #train both archiectures
            self.big.train(mode=mode)
            #train also head 
        if self.pretrained_big:
            # maybe use the weight from big in little 
            self.little.train(mode=mode)
            # train also head 
        #adapt_threshold()
        # maybe train threshold 
"""
    def pseudo_to_understand(self, threshold, delta): 
    #DELTA IS SOMETHING LIKE STEP SIZE FOR THRESHOLD 
    #threshold 
    sms = []
    for all data:
        y = self.little(x)
        score = y.topk(2)
        sm =   (score.values[0][0] - score.values[0][1])

        if (sm) < max(self.threshold, self.threshold + self.delta):
            y_big = self.big(x)
            score = y.topk(2)
            sms.append(self.beta * (score.values[0][0] - score.values[0][1]))
        else:
            sms.append(self.alpha * sm)


   
    if self.delta > 0:
        psi_delta = sum(sms)
    else:
        psi_threshold = sum(sms)
    
    if psi_threshold > psi_delta:
        self.delta *=  -0.5
    elif psi_threshold < psi_delta:
        threshold +=  delta
        delta *= 1.2
    else:
        threshold += delta
        delta += -1 

    
    # calculate PSI of threshold and PSI of threshold + delta
    # PSI = P       

"""