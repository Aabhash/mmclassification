# Copyright (c) OpenMMLab. All rights reserved.
from fvcore.nn import FlopCountAnalysis

import mmcv
import numpy as np
import torch
import torch.distributed as dist

def avg_flops(model, data_loader)
    """
        API which return the avarage Flops over a Validation Set
    """
    log = "results/BranchyNet-Imagenette/log1.txt"
    model.eval()
    prog_bar = mmcv.ProgressBar(len(dataset))
    flops = 0
    for i, data in enumerate(data_loader):
        img = data['img']
        batch_size = data['img'].size(0)
        with torch.no_grad():
            flops += FlopCountAnalysis(model, img)

    return flops / len(data_loader)
