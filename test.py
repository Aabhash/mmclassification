from mmcls.models.backbones.earlyexit import BranchyNet
import pdb
from torch import rand, max

m = BranchyNet([1, 1, 1])
t = rand(2, 3, 32, 32)

print(m(t))
