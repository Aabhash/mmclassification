from mmcls.models.backbones.earlyexit import BranchyNetImagenette2
import pdb
from torch import rand, max
from torchvision import transforms

from PIL import Image


m100 = BranchyNetImagenette2([1, 0, 0], False, [0.5, 0.5])
m110 = BranchyNetImagenette2([1, 1, 0], False,[0.5, 0.5])
m101 = BranchyNetImagenette2([1, 0, 1], False,[0.5, 0.5])
m111 = BranchyNetImagenette2([1, 1, 1], False,[0.5, 0.5])
m010 = BranchyNetImagenette2([0, 1, 0], False,[0.5, 0.5])
m001 = BranchyNetImagenette2([0, 0, 1], False,[0.5, 0.5])
m011 = BranchyNetImagenette2([0, 1, 1], False,[0.5, 0.5])

models = [(m100, "m100"), (m110, "m110"), (m101, "m101"), 
          (m111, "m111"), (m010, "m010"), (m001, "m001"), 
           (m011, "m011")]

# t = 7 * rand(8, 3, 64, 64)

img = Image.open("/home/graf-wronski/Downloads/n01440764_11400.JPEG")
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



