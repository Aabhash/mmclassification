from mmcls.models.backbones.earlyexit import BranchyNet
import pdb
from torch import rand, max

m100 = BranchyNet([1, 0, 0])
m110 = BranchyNet([1, 1, 0])
m101 = BranchyNet([1, 0, 1])
m111 = BranchyNet([1, 1, 1])
m010 = BranchyNet([0, 1, 0])
m001 = BranchyNet([0, 0, 1])
m011 = BranchyNet([0, 1, 1])

models = [(m100, "m100"), (m110, "m110"), (m101, "m101"), 
          (m111, "m111"), (m010, "m010"), (m001, "m001"), 
           (m011, "m011")]

t = 7 * rand(8, 3, 32, 32)

for pair in models:
    m = pair[0]
    print("Model: ", pair[1])

    print("m(t): \n")

    print(f"Length of: forward_train(t):{len(m.forward_train(t))} \n")
    print(f"Shape of: forward_test(t): {m.forward_test(t).shape} \n")
    print(f"forward_test(t) \n {m.forward_test(t)}")
