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

        if isinstance(pred, list):
            loss = 0.0
            for exit in pred:
                loss += branchy_net_loss(exit, target)
            return loss
        
        loss = branchy_net_loss(pred, target)
        return loss