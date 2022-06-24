from ..builder import CLASSIFIERS, build_backbone, build_head, build_neck
from ..heads import MultiLabelClsHead
from ..utils.augment import Augments
from .base import BaseClassifier
import torch

@CLASSIFIERS.register_module()
class Cascading(BaseClassifier):
    # these are also the key words for the config
    
    def __init__(self,
                 little,
                 little_head,
                 little_neck,
                 big,
                 big_neck,
                 big_head,
                 init_threshold=0.5,
                 init_delta=0.1,
                 pretrained_little = None,
                 pretrained_big = None,
                 beta = 0.85, # which denodes how much percentage sm from big is lesss important then from little
                 init_cfg=None):
        super(Cascading, self).__init__(init_cfg)
        if pretrained_little is not None:
            self.init_cfg = dict(type='Pretrained', checkpoint=pretrained_little)
        self.little = build_backbone(little)
        self.little_neck = build_neck(little_neck)
        self.little_head = build_head(little_head)
        if pretrained_big is not None:
            self.init_cfg = dict(type='Pretrained', checkpoint=pretrained_big)
        self.big = build_backbone(big)
        self.big_neck = build_neck(big_neck)
        self.big_head = build_head(big_head)
        self.threshold = init_threshold
        self.delta = init_delta
        self.beta = beta
        self.PSI = 0
        self.PSI_delta = 0
    
    def extract_feat(self, img, network, neck = None, head=None):
        x = network(img)
        if neck:
            x = neck(x)
        if head and hasattr(head, 'pre_logits'):
            x = head.pre_logits(x)
        return x


    def forward_train(self, img,   **kwargs):
        """
            normaly we don't train the the models we only train the threshold
        """
        res = self.extract_feat(img,self.little,neck=self.little_neck,head=self.little_head)
        res = self.little_head.simple_test(res)
        score = torch.tensor(res).topk(2).values.transpose(1,0)
        sm = score[0]- score[1]
        mask = torch.nonzero((sm) < max(self.threshold, self.threshold + self.delta)) #need <
        img = img[mask].squeeze()
        if img.shape[0] > 0:
            res_big = self.extract_feat(img,self.big, neck=self.big_neck, head=self.big_head)
            score_big = torch.tensor(res).topk(2).values.transpose(1,0)
            sm_big = score_big[0]- score_big[1]
            #maybe there are more elegant ways
            for (i, re) in zip(mask, res_big):
                re = re.clone()
                res[i] = re
                sm[i] = sm_big[i]
        if self.delta > 0:
            self.PSI_delta = sm.sum()
        else:
            self.PSI = sm.sum()
        if self.PSI > self.PSI_delta:
            self.delta *= -0.5
        elif self.PSI < self.PSI_delta:
            self.threshold += self.delta
        else:
            self.threshold += self.delta

        return res
    
    def simple_test(self, img,   **kwargs):
        """
            normaly we don't train the the models we only train the threshold
        """
        res = self.extract_feat(img,self.little,neck=self.little_neck,head=self.little_head)
        res = self.little_head.simple_test(res)
        score = torch.tensor(res).topk(2).values.transpose(1,0)
        sm = score[0]- score[1]
        mask = torch.nonzero((sm) < max(self.threshold, self.threshold + self.delta)) #need <
        im_shape = img.shape
        img = img[mask]
        if img.shape[0] > 0:
            #maybe this is different with different datasets
            img = img.view(img.shape[0],im_shape[1],im_shape[2],im_shape[3])
            res_big = self.extract_feat(img,self.big, neck=self.big_neck, head=self.big_head)
            res_big = self.big_head.simple_test(res_big)
            score_big = torch.tensor(res).topk(2).values.transpose(1,0)
            sm_big = score_big[0]- score_big[1]
            #maybe there are more elegant ways
            for (i, re) in zip(mask, res_big):
                res[i] = re
                sm[i] = sm_big[i]
            #res = torch.tensor(res)
        if self.delta > 0:
            self.PSI_delta = sm.sum()
        else:
            self.PSI = sm.sum()
        if self.PSI > self.PSI_delta:
            self.delta *= -0.5
        elif self.PSI < self.PSI_delta:
            self.threshold += self.delta
        else:
            self.threshold += self.delta
        
        return res
        

    #def simple_test(self, img, **kwargs):
    #  res = self.extract_feat(img,self.little,neck=self.little_neck,head=self.little_head)
    #    res = self.little_head.simple_test(res)
    #    score = torch.tensor(res).topk(2).values.transpose(1,0)
    #    mask = torch.nonzero((score[0]- score[1]) < self.threshold) #need <
    #    img = img[mask].squeeze()
    #    if img.shape[0] > 0:
    #        res_big = self.extract_feat(img,self.big, neck=self.big_neck, head=self.big_head)
    #        #maybe there are more elegant ways
    #        for (i, re) in zip(mask, res_big):
    #            re = re.clone()
    #            res[i] = re
    #    return res