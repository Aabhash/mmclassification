from math import gamma

from pyparsing import alphanums
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F    
from typing import Union, List
import pdb

from ..builder import LOSSES
from .utils import weighted_loss

# A Cross-Entropy-Loss adapted to deal with multiple Exits

@weighted_loss
def branchy_net_loss(pred: Tensor, target: Tensor) -> float:

    loss =  F.cross_entropy(pred, target)
    return loss


@LOSSES.register_module()
class BranchyNetLoss(nn.Module):

    def __init__(self):
        super(BranchyNetLoss, self).__init__()

    def forward(self, pred: Union[Tensor, List[Tensor]], target: Tensor, avg_factor=None,) -> float:
        
        '''When training BranchyNet we face one Tensor for each Exit.'''
        '''If there are multiple Tensors pred is list. Else pred is Float.'''

        loss = 0.0
        if isinstance(pred, list):
            for exit in pred:
                loss += branchy_net_loss(exit, target)
            return loss
        
        loss = branchy_net_loss(pred, target)
        return loss

@LOSSES.register_module()
class WeightedBranchyNetLoss(nn.Module):
    "A Branchy Net Loss with weights for different exits"
    def __init__(self, weights: List):
        super(WeightedBranchyNetLoss, self).__init__()
        self.weights = weights

    def forward(self, pred: Union[Tensor, List[Tensor]], target: Tensor, avg_factor=None,) -> float:
        
        '''When training BranchyNet we face one Tensor for each Exit.'''
        '''If there are multiple Tensors pred is list. Else pred is Float.'''

        loss = 0.0
        if isinstance(pred, list):
            for exit, weight in zip(pred, self.weights):
                loss += weight * branchy_net_loss(exit, target)
            return loss
        
        loss = branchy_net_loss(pred, target)
        return loss
