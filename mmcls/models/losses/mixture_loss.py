import torch
import torch.nn as nn
import torch.nn.functional as F

from ..builder import LOSSES
from .utils import weight_reduce_loss


@LOSSES.register_module()
class ExpertProbabiltityLoss(nn.Module):
    """Cross entropy loss.

    Args:
        use_sigmoid (bool): Whether the prediction uses sigmoid
            of softmax. Defaults to False.
        use_soft (bool): Whether to use the soft version of CrossEntropyLoss.
            Defaults to False.
        reduction (str): The method used to reduce the loss.
            Options are "none", "mean" and "sum". Defaults to 'mean'.
        loss_weight (float):  Weight of the loss. Defaults to 1.0.
        class_weight (List[float], optional): The weight for each class with
            shape (C), C is the number of classes. Default None.
        pos_weight (List[float], optional): The positive weight for each
            class with shape (C), C is the number of classes. Only enabled in
            BCE loss when ``use_sigmoid`` is True. Default None.
    """

    def __init__(self,
                 num_classes,
                 use_sigmoid=False,
                 use_soft=False,
                 reduction='mean',
                 loss_weight=1.0,
                 class_weight=None,
                 pos_weight=None):
        super(ExpertProbabiltityLoss, self).__init__()
        self.num_classes = num_classes

    def forward(self,
                cls_score,
                label,
                weight=None,
                avg_factor=None,
                reduction_override=None,
                **kwargs):
        label = torch.nn.functional.one_hot(label, num_classes=self.num_classes).type(torch.float32)
        label = label[:, None]
        probs = torch.bmm(label, cls_score)
        probs = torch.sum(probs, dim = 1)
        return -torch.log(probs)

        

