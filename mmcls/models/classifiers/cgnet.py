from ..builder import CLASSIFIERS, build_backbone, build_head, build_neck
from ..utils.augment import Augments
from .base import BaseClassifier
import torch


@CLASSIFIERS.register_module()
class CGClassifier(BaseClassifier):

    def __init__(self,
                 backbone,
                 neck=None,
                 head=None,
                 train_cfg=None,
                 init_cfg=None,
                 get_infos=None,
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

    def extract_feat(self, img, result_dir=None, metas=None):
        x = self.backbone(img, result_dir=result_dir, metas=metas)
        return x

    def forward_train(self, img, gt_label, **kwargs):

        if self.augments is not None:
            img, gt_label = self.augments(img, gt_label)

        x = self.extract_feat(img)

        losses = dict()
        loss = self.head.forward_train(x, gt_label)

        for name, param in self.backbone.named_parameters():
            if 'threshold' in name:
                loss['loss'] += 1e-4 * torch.sum((param - self.backbone.gtarget) ** 2)
        losses.update(loss)
        return losses

    def simple_test(self, img, img_metas=None, **kwargs):
        """Test without augmentation."""
        if self.get_infos:
            x = self.extract_feat(img, result_dir=self.get_infos, metas=img_metas)
        else:
            x = self.extract_feat(img)
        res = self.head.simple_test(x, kwargs)

        return res
