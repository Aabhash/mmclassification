from turtle import left
import torch

import torch.nn as nn
from ..builder import HEADS, build_loss
from ..utils import is_tracing
from .base_head import BaseHead
from operator import itemgetter
import json
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
        # self.compute_accuracy = Accuracy(topk=self.topk)
        self.softmax = nn.Softmax(dim=1).cuda()
        self.simple_softmax = nn.Softmax(dim=0).cuda()

        # self.T = torch.Tensor([0.65] * (num_exits-1) + [0])
        self.T = torch.hstack((torch.linspace(0.8, 0.5, (num_exits-1)), torch.tensor([0])))

        self.exit_tracker = {ncls:[] for ncls in range(num_exits)}
        self.exits = {ncls:0 for ncls in range(num_exits)}
        self.total_time = 0


    def write_infos(self, result_file):
        with open(result_file, "w+") as f:
            json.dump(self.exit_tracker, f)


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


    def simple_test(self, x, post_process=True, result_file=None, **kwargs):
        # st = time.time()
        # if isinstance(x, tuple):
            # x = x[-1]

        metas = kwargs.get("img_metas", None)
        if result_file:
            metas = [m['ori_filename'] for m in metas]

        pred = torch.zeros_like(x[-1])
        left_to_track_idx = torch.arange(x[-1].shape[0])

        for k, out in enumerate(x):
            if left_to_track_idx.numel() > 0:
                # try:
                logits = self.softmax(out[left_to_track_idx])
                max_preds, max_idx = torch.max(logits, dim=1)
                # Indices with val > Threshold
                curr_idx = max_preds.ge(self.T[k]).nonzero(as_tuple=True)
                og_idx = left_to_track_idx[curr_idx]
                left_to_track_idx = left_to_track_idx[max_preds.le(self.T[k]).nonzero(as_tuple=True)]

                pred[og_idx] = logits[curr_idx]
                for id in og_idx:
                    self.exit_tracker[k].append(metas[id])
                    self.exits[k] += 1
                # try:
                    # self.exit_tracker[k].append(itemgetter(*og_idx)(metas))
                # except TypeError:
                    # pass
            else:
                break
    
        if result_file:
            self.write_infos(result_file)

        # pred = []
        # for k, out in enumerate(x):
        #     logits = self.softmax(out)
        #     pred.append(logits)

        # self.total_time += (time.time() - st)
        # print(self.total_time, self.exits)

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
