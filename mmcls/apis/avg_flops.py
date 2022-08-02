# Copyright (c) OpenMMLab. All rights reserved.
from fvcore.nn import FlopCountAnalysis

import mmcv
import numpy as np
import torch
import torch.distributed as dist
import pdb


def avg_flops(model, data_loader):
    """
        API which return the avarage Flops over a Validation Set
    """
    model.eval()
    flops = 0
    
    for i, data in enumerate(data_loader):
        img = data['img']
        batch_size = img.size(0)
        with torch.no_grad():
            flops += FlopCountAnalysis(model.cuda(), (img.cuda(),False)).total()
        if i % 40 == 0:
            print(f"Batch {i} done")

    return flops / len(data_loader.dataset)


