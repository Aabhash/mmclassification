# Copyright (c) OpenMMLab. All rights reserved.
from fvcore.nn import FlopCountAnalysis

import mmcv
import numpy as np
import torch
import torch.distributed as dist
import pdb

import sys, os


def avg_flops(model, data_loader):
    """
        API which return the avarage Flops over a Validation Set
    """
    model.eval()
    flops = 0
    blockPrint() # we have to block print because of FlopCountAnalysis Output
    for i, data in enumerate(data_loader):
        img = data['img']
        batch_size = img.size(0)
        with torch.no_grad():
            flops += FlopCountAnalysis(model, img).total()
        if i % 40 == 0:
            enablePrint()
            print(f"Batch {i} done")
            blockPrint()

    enablePrint()

    return flops / len(data_loader.dataset)


# Disable
def blockPrint():
    sys.stdout = open(os.devnull, 'w')

# Restore
def enablePrint():
    sys.stdout = sys.__stdout__
