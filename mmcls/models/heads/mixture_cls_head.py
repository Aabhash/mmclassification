from numpy import argmax
from ..builder import HEADS
from typing import Dict, Sequence

import torch
import torch.nn as nn
from mmcv.cnn import build_activation_layer, build_norm_layer
from mmcv.runner import BaseModule, ModuleList
from .cls_head import ClsHead
from torch.nn import Identity


@HEADS.register_module()
class MixtureClsHead(ClsHead):

    def __init__(self,
                 num_classes,
                 in_channels,
                 in_gate_head,
                 loss=dict(type='CrossEntropyLoss', loss_weight=1.0),
                 topk=(1,),
                 n_experts=5):
        super(MixtureClsHead, self).__init__(loss=loss, topk=topk)
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.in_gate_head = in_gate_head
        self.n_experts = n_experts
        self._init_layers()

    def _init_layers(self):
        self.experts = ModuleList()
        for _ in range(self.n_experts):
            self.experts.append(nn.Linear(self.in_channels, self.num_classes))
        # maybe 192 should be a variable
        self.gate = nn.Linear(self.in_gate_head, self.n_experts)

    def pre_logits(self, x):
        if isinstance(x, tuple):
            x = x[-1]
        return x

    def simple_test(self, values, gates, softmax=True, post_process=True):
        gates = self.pre_logits(gates)
        gates = torch.tensor(self.gate(gates))
        gates = gates[:, :, None]
        #here could be an issue
        values = torch.stack([torch.tensor(self.experts[i](
            self.pre_logits(values[i])))for i in range(self.n_experts)], axis=-1)
        values, gates = nn.functional.softmax(
            values, dim = 1), nn.functional.softmax(gates, dim = 1)
        x = torch.matmul(values, gates)
        #x = nn.functional.softmax(x)
        #y = torch.argmax(x, dim = 1)
        return x

    def forward_train(self, values, gates,  gt_label):
        gates = self.pre_logits(gates)
        gates = self.gate(gates)
        gates = gates[:, :, None]

        values = torch.stack([torch.tensor(self.experts[i](
            self.pre_logits(values[i])))for i in range(self.n_experts)], axis=-1)
        values, gates = nn.functional.softmax(
            values, dim = 1), nn.functional.softmax(gates,dim = 1 )
        x = torch.matmul(values, gates)
        #x = nn.functional.softmax(x)
        #x = x[:,None]
        losses = self.loss(x, gt_label)
        return losses
