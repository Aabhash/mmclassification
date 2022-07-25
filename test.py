from mmcls.models.backbones.earlyexit import BranchyNetImagenette
import pdb
from torch import rand, max
from torchvision import transforms

from PIL import Image


m100 = BranchyNetImagenette([1, 0, 0])
m110 = BranchyNetImagenette([1, 1, 0])
m101 = BranchyNetImagenette([1, 0, 1])
m111 = BranchyNetImagenette([1, 1, 1])
m010 = BranchyNetImagenette([0, 1, 0])
m001 = BranchyNetImagenette([0, 0, 1])
m011 = BranchyNetImagenette([0, 1, 1])

models = [(m100, "m100"), (m110, "m110"), (m101, "m101"), 
          (m111, "m111"), (m010, "m010"), (m001, "m001"), 
           (m011, "m011")]

# t = 7 * rand(8, 3, 64, 64)

img = Image.open("/home/carlwanninger/till/mmclassification/data/imagenette2/val/n01440764/n01440764_11400.JPEG")
convert_tensor = transforms.ToTensor()
t = convert_tensor(img)[None, :] # we need to add a dummy dimension as batch size 

# pdb.set_trace()

for pair in models:
    m = pair[0]
    print("Model: ", pair[1])

    print("m(t): \n")

    print(f"Length of: forward_train(t):{len(m.forward_train(t))} \n")
    print(f"Shape of: forward_test(t): {m.forward_test(t).shape} \n")
    print(f"forward_test(t) \n {m.forward_test(t)}")



