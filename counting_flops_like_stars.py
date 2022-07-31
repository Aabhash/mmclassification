from pickle import BININT
from tkinter import image_names

from sklearn.ensemble import GradientBoostingClassifier
from mmcls.models.backbones.earlyexit import BranchyNetImagenette2, BranchyNet
from tools.analysis_tools.flop_counter import *
from mmcls.models.backbones.grgbnet import GRGBnet_Base
from torch import rand, cuda

from fvcore.nn import FlopCountAnalysis

# Building Models and setting Input variables

if cuda.is_available():
    device = 'cuda'
else:
    device = 'cpu'

BNCF = BranchyNet(activated_branches=[True, True, True])
BNI = BranchyNetImagenette2(activated_branches=[True, True, True])

BNI_list = [BNI]
BNCF_list = [BNCF]

GRGB_RGB = GRGBnet_Base(use_grayscale=False, use_rgb=True)
GRGB_G = GRGBnet_Base(use_grayscale=True, use_rgb=False)
GRGB = GRGBnet_Base(use_grayscale=True, use_rgb=True)

GRGB_list = [GRGB_G, GRGB_RGB, GRGB]

model_list = BNI_list + BNCF_list + GRGB_list

for m in model_list:
    m = m.to(device)


H_Imagenette = 224
W_Imagenette = 224

H_Cifar = 32
W_Cifar = 32

random_Imagenette = rand(1, 3, H_Imagenette, W_Imagenette).to(device)
random_Imagenette_batch = rand(16, 3, H_Imagenette, W_Imagenette).to(device)

random_Cifar = rand(1, 3, H_Cifar, W_Cifar).to(device)
random_Cifar_batch = rand(16, 3, H_Cifar, W_Cifar).to(device)

# ------------------- TESTING -------------------
# flop_counter.py 

print("Flop Count Analysis with flop_counter.py ")

print("\n Total Flops for GRGB on Imagenette - RGB only")
measure_model(GRGB_RGB, H_Imagenette, W_Imagenette)

print("\n Total Flops for GRGB on Imagenette - Greyscale only")
measure_model(GRGB_G, H_Imagenette, W_Imagenette)

print("\n Flops for GRGB on Imagenette")
measure_model(GRGB, H_Imagenette, W_Imagenette)

# Flop Count Analysis

print(" _______________________________________ \n")
print("Flop Count Analysis with FlopCountAnalysis")
print("GRGB-Net")
print("Using Random Tensor")

print("\n RGB only")
flops = FlopCountAnalysis(GRGB_RGB, random_Imagenette)
print(f"\n Total Flops: ", flops.total())
print(f"\n Flops by Operator:", flops.by_operator())

print("\n Greyscale only")
flops = FlopCountAnalysis(GRGB_G, random_Imagenette)
print(f"\n Total Flops: ", flops.total())
print(f"\n Flops by Operator:", flops.by_operator())

print("\n GRGB-Net")
flops = FlopCountAnalysis(GRGB, random_Imagenette)
print(f"\n Total Flops: ", flops.total())
print(f"\n Flops by Operator:", flops.by_operator())

print("")

print(" _______________________________________ \n")
print("Flop Count Analysis with FlopCountAnalysis")
print("BranchyNet-Net")
print("Using Random Tensor Batch")
print("Imagenette")

flops = FlopCountAnalysis(BNI, random_Imagenette_batch)
print(f"\n Total Flops: ", flops.total())
print(f"\n Flops by Operator:", flops.by_operator())

print(" _______________________________________ \n")
print("Flop Count Analysis with FlopCountAnalysis")
print("BranchyNet-Net")
print("Using Random Tensor Batch")
print("Cifar-10")

flops = FlopCountAnalysis(BNCF, random_Cifar_batch)
print(f"\n Total Flops: ", flops.total())
print(f"\n Flops by Operator:", flops.by_operator())
