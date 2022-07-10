from ..builder import HEADS
from .cls_head import ClsHead
from torch.nn import Identity

@HEADS.register_module()

class emptyClsHead(ClsHead):

    def __init__(self, loss=dict(type='CrossEntropyLoss', loss_weight=1.0), topk=(1,)):
        super(emptyClsHead, self).__init__(loss=loss, topk=topk)

        self._init_layers()

    def _init_layers(self):
        self.id = Identity()

    def forward_train(self, x, gt_label):
        cls_score = self.id(x)
        losses = self.loss(cls_score, gt_label)
        return losses
