from turtle import left
import torch

import torch.nn as nn
from ..builder import HEADS, build_loss
from ..utils import is_tracing
from .base_head import BaseHead
from mmcls.models.losses import Accuracy
import time

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
                 num_exits=6,
                 init_cfg=None):
        super().__init__(init_cfg=init_cfg)

        assert isinstance(loss, dict)

        self.topk = topk
        self.compute_loss = build_loss(loss)
        self.compute_accuracy = Accuracy(topk=self.topk)
        self.softmax = nn.Softmax(dim=1).cuda()
        self.simple_softmax = nn.Softmax(dim=0).cuda()
        self.T = torch.Tensor([0.5] * (num_exits-1) + [0])
        self.exit_tracker = {ncls:0 for ncls in range(num_exits)}
        self.total_time = 0

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

        # accs = [self.compute_accuracy(cls_scores[i], gt_label) for i in range(len(cls_scores))]

        # losses['accuracy'] = {}
        # for i, acc in enumerate(accs):
            # for c, k in enumerate(self.topk):
                # losses['accuracy'][f'cls{i}-top{k}-accuracy']= acc[c]

        return losses

    def forward_train(self, cls_score, gt_label, **kwargs):
        # if isinstance(cls_score, tuple):
        #     cls_score = cls_score[-1]
        # gt_label = gt_label.type_as(cls_score)
        return self.loss(cls_score, gt_label, **kwargs)


    def simple_test(self, x, post_process=True, **kwargs):
        # st = time.time()
        # if isinstance(x, tuple):
            # x = x[-1]

        pred = torch.zeros_like(x[-1])
        left_to_track_idx = torch.arange(x[-1].shape[0])

        for k, out in enumerate(x):
            if left_to_track_idx.numel() > 0:
                try:
                    logits = self.softmax(out[left_to_track_idx])
                    max_preds, max_idx = torch.max(logits, dim=1)
                    # Indices with val > Threshold
                    curr_idx = max_preds.ge(self.T[k]).nonzero().squeeze()
                    og_idx = left_to_track_idx[curr_idx]
                    left_to_track_idx = left_to_track_idx[max_preds.le(self.T[k]).nonzero().squeeze()]
         
                    pred[og_idx] = logits[curr_idx]
                    self.exit_tracker[k] += curr_idx.numel()
                except IndexError:
                    pred[left_to_track_idx] = self.simple_softmax(out[left_to_track_idx])
                    self.exit_tracker[k] += 1
            else:
                break

        # pred = []
        # for k, out in enumerate(x):
        #     logits = self.softmax(out)
        #     pred.append(logits)

        # self.total_time += (time.time() - st)
        # print(self.total_time, self.exit_tracker)

        if post_process:
            return self.post_process(pred)
        else:
            return pred

    def post_process(self, pred):
        on_trace = is_tracing()
        if torch.onnx.is_in_onnx_export() or on_trace:
            return pred
        # pred = [list(p.detach().cpu().numpy()) for p in pred]
        pred = pred.detach().cpu().numpy()
        return pred
