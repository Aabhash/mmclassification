from ..builder import CLASSIFIERS, build_backbone, build_head, build_neck
from ..utils.augment import Augments
from .base import BaseClassifier
import torch


@CLASSIFIERS.register_module()
class MultiScaleClassifier(BaseClassifier):

    def __init__(self,
                 backbone,
                 neck=None,
                 head=None,
                 train_cfg=None,
                 get_infos=None,
                 init_cfg=None,
                 pretrained=None):
        super().__init__(init_cfg)

        self.get_infos = get_infos
        self.backbone = build_backbone(backbone)
        self.n_classifiers = 0

        if neck is not None:
            self.neck = build_neck(neck)

        if head is not None:
            self.head = build_head(head)

        self.augments = None
        if train_cfg is not None:
            augments_cfg = train_cfg.get('augments', None)
            if augments_cfg is not None:
                self.augments = Augments(augments_cfg)

    def extract_feat(self, img):
        x = self.backbone(img)
        self.n_classifiers = len(x)
        return x

    def forward_train(self, img, gt_label, **kwargs):

        if self.augments is not None:
            img, gt_label = self.augments(img, gt_label)

        x = self.extract_feat(img)

        if not isinstance(x, list):
            x = [x]

        losses = dict()
        loss = self.head.forward_train(x, gt_label)
        losses.update(loss)
        return losses

    def simple_test(self, img, **kwargs):
        """Test without augmentation."""
        x = self.extract_feat(img)
        res = self.head.simple_test(x, result_file=self.get_infos, **kwargs)

        return res
