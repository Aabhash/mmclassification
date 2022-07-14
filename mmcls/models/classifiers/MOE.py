from tkinter import Y
from matplotlib.pyplot import axis
from ..builder import CLASSIFIERS, build_backbone, build_head, build_neck
from ..utils.augment import Augments
from .base import BaseClassifier
import torch
import numpy as np

#m = model.MOE_Model(model.get_experts(cmd.n_experts, input_shape), model.get_gate(cmd.n_experts, input_shape))


@CLASSIFIERS.register_module()
class MOE(BaseClassifier):
    # these are also the key words for the config
    def __init__(self,
                 expert_backbone,
                 expert_neck, 
                 gate_network,
                 gate_neck,
                 end_head,
                 n_experts=5,
                 init_cfg=None):
        super(MOE, self).__init__(init_cfg)
        self.experts_backbone = torch.nn.ModuleList()
        self.experts_neck = torch.nn.ModuleList()
        self.gate = build_backbone(gate_network)
        self.gate_neck = build_neck(gate_neck)
        self.n_experts = n_experts
        self.end_head = build_head(end_head)
        for _ in range(self.n_experts):
            # which backbone I have to configure
            self.experts_backbone.append(build_backbone(expert_backbone))
            # if neck is not trainable 1 is enough
            self.experts_neck.append(build_neck(expert_neck))

    def extract_feat(self, img):
        gates = self.gate_neck(self.gate(img))
        values = [self.experts_neck[i]
        (self.experts_backbone[i](img))for i in range(self.n_experts)]
        return values, gates

    
    def forward_train(self, img, gt_label, **kwargs):
        """

        """
        # if self.augments is not None:
        #    img, gt_label = self.augments(img, gt_label)

        values, gates  = self.extract_feat(img)
        
        losses = dict()
        
        # {'loss': tensor(2.3073, devic...ackward0>)}
        # 
        loss = self.end_head.forward_train(values, gates, gt_label)
        
        
        # loss function I don't now how this work but i figure it out
        losses.update(loss)
        return losses

    def simple_test(self, img, **kwargs):
        """

        """
        values, gates  = self.extract_feat(img)
        y = self.end_head.simple_test(values, gates)
        return y
