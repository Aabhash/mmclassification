from mmcls.models.backbones.earlyexit import BranchyNetImagenette
import pdb
from torch import rand, max, device, cuda
from torchvision import transforms
import torch.nn as nn
from PIL import Image


mydevice = device("cuda" if cuda.is_available() else "cpu")

#m100 = BranchyNetImagenette([1, 0, 0])
#m110 = BranchyNetImagenette([1, 1, 0])
#m101 = BranchyNetImagenette([1, 0, 1])
m111 = BranchyNetImagenette([1, 1, 1]).to(mydevice)
#m010 = BranchyNetImagenette([0, 1, 0])
#m001 = BranchyNetImagenette([0, 0, 1])
#m011 = BranchyNetImagenette([0, 1, 1])

#models = [(m100, "m100"), (m110, "m110"), (m101, "m101"), 
#          (m111, "m111"), (m010, "m010"), (m001, "m001"), 
#           (m011, "m011")]

t = rand(16, 3, 224, 224) # output of layer2

# img = Image.open("/home/carlwanninger/till/mmclassification/data/imagenette2/val/n01440764/n01440764_11400.JPEG")
# convert_tensor = transforms.ToTensor()
# t = convert_tensor(img)[None, :] # we need to add a dummy dimension as batch size 

# pdb.set_trace()

# x = nn.Conv2d(512, 1024, 7, 3)(t)
# x = nn.BatchNorm2d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)(x)
# x = nn.ReLU()(x)
# x = nn.Conv2d(1024, 2048, 5, 3)(x)
#x = nn.BatchNorm2d(2048, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)(x)
#x  = nn.ReLU()(x) 
#x = nn.Conv2d(2048, 4096, 5, 3)(x)
#x = nn.BatchNorm2d(4096, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)(x)
#x = nn.ReLU()(x)
#x = nn.Conv2d(4096, 4096, 3, 2, padding=1)(x)
#x = nn.BatchNorm2d(4096, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)(x)
#x = nn.ReLU()(x)
#x = nn.AvgPool2d(3, stride=3, padding=0)(x)
#x = nn.Flatten()(x)
#x = nn.Linear(8192, 2048)(x)
#x = nn.Linear(2048, 10)(x)
#x = nn.Softmax(dim=1)(x)

print(m111(t, return_loss=True))

# for pair in models:
#     m = pair[0]
#     print("Model: ", pair[1])

#     print("m(t): \n")

#     print(f"Length of: forward_train(t):{len(m.forward_train(t))} \n")
#     print(f"Shape of: forward_test(t): {m.forward_test(t).shape} \n")
#     print(f"forward_test(t) \n {m.forward_test(t)}")



