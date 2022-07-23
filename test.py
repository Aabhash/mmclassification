from mmcls.models.backbones.earlyexit import BranchyNet
import pdb
from torch import rand, max

m = BranchyNet([1, 1, 1])
t = 7 * rand(8, 3, 32, 32)

print("m(t): \n")
print(m(t, return_loss = False), "\n")

print(f"Length of: forward_train(t):{len(m.forward_train(t))} \n")
print(f"Shape of: forward_test(t): {m.forward_test(t).shape} \n")
print(f"forward_test(t) \n {m.forward_test(t)}")
