import torch

import torch.nn as nn
from ..builder import HEADS, build_loss
from ..utils import is_tracing
from .base_head import BaseHead
from mmcls.models.losses import Accuracy


@HEADS.register_module()
class MultiScaleHead(BaseHead):

    def __init__(self,
                 loss=dict(
                     type='CrossEntropyLoss',
                     use_sigmoid=True,
                     reduction='mean',
                     loss_weight=1.0
                 ),
                 topk=(1,),
                 init_cfg=None):
        super().__init__(init_cfg=init_cfg)

        assert isinstance(loss, dict)

        self.topk = topk
        self.compute_loss = build_loss(loss)
        self.compute_accuracy = Accuracy(topk=self.topk)
        self.softmax = nn.Softmax(dim=1).cuda()
        self.T = torch.Tensor([0.5, 0.5, 0.5, 0.5, 0])

    def loss(self, cls_scores, gt_label):
        # gt_label = gt_label.type_as(cls_score)
        # num_samples = len(cls_score)
        losses = dict()

        # _gt_label = torch.abs(gt_label)
        
        # Calculate loss
        loss = 0.0
        for score in cls_scores:
            loss += self.compute_loss(score, gt_label)

        # Calculate accuracy of the last classifier
        acc = self.compute_accuracy(cls_scores[-1], gt_label)
        assert len(acc) == len(self.topk)

        losses['loss'] = loss
        losses['accuracy'] = {
            f'top-{k}': a for k, a in zip(self.topk, acc)
        }
        return losses

    def forward_train(self, cls_score, gt_label, **kwargs):
        # if isinstance(cls_score, tuple):
        #     cls_score = cls_score[-1]
        # gt_label = gt_label.type_as(cls_score)
        return self.loss(cls_score, gt_label, **kwargs)


    def simple_test(self, x, post_process=False, **kwargs):
        
        if isinstance(x, tuple):
            x = x[-1]

        pred = []

        for i, out in enumerate(x):
            logits = self.softmax(out)
            max_preds, max_idx = torch.max(logits, dim=1)
            exits = max_preds.ge(self.T[i])
            breakpoint()
            # for j in range()

        if post_process:
            return self.post_process(pred)
        else:
            return pred

    def post_process(self, pred):
        on_trace = is_tracing()
        if torch.onnx.is_in_onnx_export() or on_trace:
            return pred
        pred = list(pred.detach().cpu().numpy())
        return pred
