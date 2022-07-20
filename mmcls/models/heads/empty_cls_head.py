from ..builder import HEADS
from .cls_head import ClsHead
from torch.nn import Identity
import pdb

@HEADS.register_module()

class emptyClsHead(ClsHead):

    def __init__(self, loss=dict(type='BranchyNetLoss', loss_weight=1.0), topk=(1,)):
        super(emptyClsHead, self).__init__(loss=loss, topk=topk)

        self._init_layers()

    def _init_layers(self):
        self.id = Identity()

    def forward_train(self, x, gt_label):
        # print("HEAD")
        # pdb.set_trace()
        cls_score = self.id(x)
        losses = self.loss(cls_score, gt_label)
        return losses
