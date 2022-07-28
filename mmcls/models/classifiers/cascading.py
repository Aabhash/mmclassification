from cv2 import threshold
from importlib_metadata import pass_none
from ..builder import CLASSIFIERS, build_backbone, build_head, build_neck
from ..heads import MultiLabelClsHead
from ..utils.augment import Augments
from .base import BaseClassifier
import torch
import numpy as np
import time 




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
                 get_infos = None,
                 beta = 0.85, # which denodes how much percentage sm from big is lesss important then from little
                 init_cfg=None,
                 train_cfg=None):
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
        self.augments = None
        self.get_infos = get_infos
        if train_cfg is not None:
            augments_cfg = train_cfg.get('augments', None)
            if augments_cfg is not None:
                self.augments = Augments(augments_cfg)
    
    def extract_feat(self, img, network, neck = None, head=None):
        x = network(img)
        if neck:
            x = neck(x)
        if head and hasattr(head, 'pre_logits'):
            x = head.pre_logits(x)
        return x


    def forward_train(self, img, gt_label,   **kwargs):
        """
            normaly we don't train the the models
        """
        
        if self.augments is not None:
            img, gt_label = self.augments(img, gt_label)

        x_1 = self.extract_feat(img,self.little,neck=self.little_neck,head=self.little_head)

        losses = dict()
        loss_1 = self.little_head.forward_train(x_1, gt_label)

        losses.update(loss_1)
        if self.augments is not None:
            img, gt_label = self.augments(img, gt_label)

        x_2 = self.extract_feat(img,self.big,neck=self.big_neck,head=self.big_head)

        losses = dict()
        loss_2 = self.big_head.forward_train(x_2, gt_label)

        losses.update(loss_2)

        return losses    

    def writeInfos(self, mask, get_infos,  kwargs):
        heavy = ""
        for i in mask:
          heavy += kwargs["img_metas"][int(i)]["filename"] + "\n"
        f = open(get_infos, "a")
        f.write(heavy)
        f.close()

    
    def simple_test(self, img, **kwargs):
        """
            normaly we don't train the the models we only train the threshold
        """
        res = self.extract_feat(img,self.little,neck=self.little_neck,head=self.little_head)
        res = self.little_head.simple_test(res)
        score = torch.tensor(np.array(res)).topk(2).values.transpose(1,0)
        sm = score[0]- score[1]
        mask_1 = torch.nonzero((sm) < self.threshold) #need <
       
        im_shape = img.shape
        img_max = img[mask_1]
        sm_max = sm
        res_threshold = res
        if img_max.shape[0] > 0:
            
            #maybe this is different with different datasets
            img_max = img_max.view(img_max.shape[0],im_shape[1],im_shape[2],im_shape[3])
            res_big = self.extract_feat(img_max,self.big, neck=self.big_neck, head=self.big_head)
            res_big = self.big_head.simple_test(res_big)
            score_big = torch.tensor(np.array(res)).topk(2).values.transpose(1,0)
            sm_big = score_big[0]- score_big[1]
            #maybe there are more elegant ways
            
            for (i, re) in zip(mask_1, res_big):
                res_threshold[i] = re
                sm_max[i] = sm_big[i]
        mask_2 = torch.nonzero((sm) < (self.threshold + self.delta))
        img_min = img[mask_2]
        sm_min = sm
        res_delta = res
        if img_min.shape[0] > 0:
            
            #maybe this is different with different datasets
            img_min = img_min.view(img_min.shape[0],im_shape[1],im_shape[2],im_shape[3])
            res_big = self.extract_feat(img_min,self.big, neck=self.big_neck, head=self.big_head)
            res_big = self.big_head.simple_test(res_big)
            score_big = torch.tensor(np.array(res)).topk(2).values.transpose(1,0)
            sm_big = score_big[0]- score_big[1]
            #maybe there are more elegant ways
            
            for (i, re) in zip(mask_2, res_big):
                res_delta[i] = re
                sm_min[i] = sm_big[i]
        res = res_delta if self.delta > 0 else  res_threshold
            #res = torch.tensor(res)
        #if self.get_infos:
        mask = mask_2 if self.delta > 0 else  mask_1
        self.writeInfos(mask, self.get_infos, kwargs)

        if self.threshold > 1 and self.PSI <= self.PSI_delta and self.delta > 0: 
            return res 
            #because over 1 is not possible 
      
        self.PSI_delta = sm_max.sum()
        self.PSI = sm_min.sum()
        if self.PSI > self.PSI_delta:
           
            self.delta = self.delta * -0.5
        if self.PSI < self.PSI_delta:
           
            self.threshold = self.threshold + self.delta
            self.delta = 1.2 * self.delta
        else:
           
            self.threshold = self.threshold + self.delta
            self.delta = -self.delta
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