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

        losses['loss'] = loss

        # Calculate accuracy of the last classifier
        # acc = self.compute_accuracy(cls_scores[-1], gt_label)
        accs = [self.compute_accuracy(cls_scores[i], gt_label) for i in range(len(cls_scores))]

        # assert len(acc) == len(self.topk)
        losses['accuracy'] = {}
        for i, acc in enumerate(accs):
            for c, k in enumerate(self.topk):
                losses['accuracy'][f'cls{i}-top{k}-accuracy']= acc[c]
        # losses['accuracy'] = {
        #     f'accuracy-exit-{i}-top{k}': a for i, a in enumerate(accs)
        # }
        return losses

    def forward_train(self, cls_score, gt_label, **kwargs):
        # if isinstance(cls_score, tuple):
        #     cls_score = cls_score[-1]
        # gt_label = gt_label.type_as(cls_score)
        return self.loss(cls_score, gt_label, **kwargs)


    def simple_test(self, x, post_process=True, **kwargs):
        
        if isinstance(x, tuple):
            x = x[-1]
        pred = []

        # exit_tracker = {nlabel:[] for nlabel in range(10)}
        
        for i, out in enumerate(x):
            logits = self.softmax(out)
            pred.append(logits)

            # max_preds, max_idx = torch.max(logits, dim=1)
            # exits = max_preds.ge(self.T[i])
            # # for j, pred_gt in enumerate(exits):
            # #     if pred_gt:
            # #         pred.append()
            # #         max_preds
                    
                    
        # pred = self.softmax(x[0])
        if post_process:
            return self.post_process(pred)
        else:
            return pred[-1]

    def post_process_single(self, pred):
        on_trace = is_tracing()
        if torch.onnx.is_in_onnx_export() or on_trace:
            return pred
        pred = list(pred.detach().cpu().numpy())
        return pred
        
    def post_process(self, pred):
        on_trace = is_tracing()
        if torch.onnx.is_in_onnx_export() or on_trace:
            return pred
        pred = [list(p.detach().cpu().numpy()) for p in pred]
        return pred[-1]
