# Copyright (c) Carl. All rights reserved.
from torch import Tensor, zeros, ones, device
import torch.nn as nn
from torch.hub import load_state_dict_from_url
from torchvision.models.resnet import ResNet, Bottleneck
from torch import load, save, sum, max # WIP: Not needed in final version I guess
from pathlib import Path
from collections import OrderedDict

import sys
import pdb

# sys.path.append('/home/graf-wronski/Projects/dynamic-networks/openmllab/mmclassification_private/mmcls/models/backbones/')
# sys.path.append('/home/graf-wronski/Projects/dynamic-networks/openmllab/mmclassification_private/mmcls/models/')

from ..builder import BACKBONES
from .base_backbone import BaseBackbone
from . import ResNet_CIFAR

# @BACKBONES.register_module()

# class res_inter_classifiers(nn.Module):
#     # ToDo: Add input information
    
#     """`A resnet with intermediate classifiers backbone.

#     Args:
#         path_resnet: The path where the base resnet can be found
#     """

#     def __init__(self, path_resnet: str = None):

#         super(res_inter_classifiers, self).__init__()

#         print(f"Path_Resnet: {path_resnet}")

#         # super(ResNet, self).__init__()

#         self.model = ResNet_CIFAR(depth=50)
#         # self.model2 = ResNet(block=Bottleneck, layers = [3, 4, 6, 3], num_classes=10)

#         # ResNet will be pretrained in the project group at one point
#         # Alternative: Get ResNet from the web

#         if path_resnet == None:

#             dirname = Path(__file__).parent.parent.parent.parent
#             print(dirname)
#             resnet_path_backbone = dirname /  'work_dirs/resnet50cifar10_backbone.pth'
#             # model_path = dirname /  'results_early_exit/checkpoints/EE-resnet50-pytorch.pth'

#             print(f"resnet_path_backbone: {resnet_path_backbone}")

#             if (resnet_path_backbone).is_file():
                
#                 state_dict = load(resnet_path_backbone)
            
#                 state_dict = OrderedDict([(k.replace("backbone.", "").replace("head.", ""),v) for k,v in state_dict.items()])

#                 self.model.fc = nn.Linear(2048, 10, bias=True)
#                 self.model.load_state_dict(state_dict)

#                 print(f"{resnet_path_backbone} used.")

#             save(self.model.state_dict(), resnet_path_backbone)

#             self.layer1 = nn.Sequential(
#                 self.model.conv1,
#                 self.model.bn1,
#                 self.model.relu,
#                 self.model.layer1
#             )

#             self.earlyExit1 = nn.Sequential(
#                 nn.Conv2d(256, 512, 5, 3),
#                 nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
#                 nn.ReLU(),
#                 nn.Conv2d(512, 1024, 5, 2),
#                 nn.BatchNorm2d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
#                 nn.ReLU(),
#                 nn.Conv2d(1024, 2048, 3, 2),
#                 nn.BatchNorm2d(2048, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
#                 nn.ReLU(),
#                 nn.AvgPool2d(3, stride=2, padding=1),
#                 nn.Flatten(),
#                 self.model.fc,
#                 nn.Softmax(dim=1)
#             )

#             self.layer2 = self.model.layer2

#             # self.early_exit_2 = 

#             self.layer3 = self.model.layer3

#             # self.early_exit_3 = 

#             self.layer4 = nn.Sequential(
#                 self.model.layer4,
#                 nn.AvgPool2d(3, stride=2, padding=1),
#                 nn.Flatten(),
#                 nn.Linear(8192, 2048),
#                 self.model.fc,
#                 nn.Softmax(dim=1)
#             )

#             # if (model_path).is_file():
#             #     state_dict = load(model_path)
#             #     self.model.load_state_dict(state_dict)
#             #     print(f"{model_path} used.")

#         # self.num_classes = num_classes   
        
#     def forward(self, img, return_loss=False):
#         if return_loss:
#             return self.forward_train(img)
#         else:
#             return self.forward_test(img)

#     def forward_train(self, x):
        
#         x = self.layer1(x)
        
#         y1 = self.earlyExit1(x)

#         # [1, 256, H/4, W/4]
#         x = self.layer2(x)
#         # [1, 512,  H/8, W/8]
#         x = self.layer3(x)
#         # [1, 1024, H/16, W/16]

#         y2 = self.layer4(x)
#         # [1, 10]
#         return y1
#         return 0.5 * y1 + 0.5 * y2

#     def forward_test(self, x):

#             x = self.layer1(x)

#             y1 = self.earlyExit1(x)

#             if max(y1) > 0.8:
#                 return y1

#             # [1, 256, H/4, W/4]
#             x = self.layer2(x)
#             # [1, 512,  H/8, W/8]
#             x = self.layer3(x)
#             # [1, 1024, H/16, W/16]

#             y2 = self.layer4(x)
#             # [1, 10]
#             return y1


@BACKBONES.register_module()
class BranchyNet(nn.Module):
    
    """ The BaseBlock of my BranchyNet Version. 
        It contains the first layer and the first exit.
        It also keeps hold of the whole loaded ResNet.

    """

    def __init__(self, activated_branches: list):

        super(BranchyNet, self).__init__()

        # The variable activated branches stores which Branches to use during Test Phase
        assert(any(activated_branches))
        self.activated_branches = activated_branches.copy()

        self.model = ResNet_CIFAR(depth=50)
        self.model.fc = nn.Linear(2048, 10, bias=True)

        # Load Pretrained Resnet 

        dirname = Path(__file__).parent.parent.parent.parent
        resnet_path_backbone = dirname /  'work_dirs/resnet50cifar10_backbone.pth'

        if not (resnet_path_backbone.is_file()):
            sys.exit(f"class exitOne requieres pretrained resNet. {resnet_path_backbone} is no file.")
        
        state_dict = load(resnet_path_backbone)
        state_dict = OrderedDict([(k.replace("backbone.", "").replace("head.", ""),v) for k,v in state_dict.items()])

        self.model.load_state_dict(state_dict)
        # save(self.model.state_dict(), resnet_path_backbone)

        self.layer1 = nn.Sequential(
            self.model.conv1,
            self.model.bn1,
            self.model.relu,
            self.model.layer1
        )

        self.earlyExit1 = nn.Sequential(
            nn.Conv2d(256, 512, 5, 3),
            nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.ReLU(),
            nn.Conv2d(512, 1024, 5, 2),
            nn.BatchNorm2d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.ReLU(),
            nn.Conv2d(1024, 2048, 3, 2),
            nn.BatchNorm2d(2048, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.ReLU(),
            nn.AvgPool2d(3, stride=2, padding=1),
            nn.Flatten(),
            self.model.fc,
            nn.Softmax(dim=1)
        )

        self.layer2 = self.model.layer2

        self.layer3 = self.model.layer3

        self.earlyExit2 = nn.Sequential(
            nn.Conv2d(1024, 2048, 3, 2),
            nn.BatchNorm2d(2048, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.ReLU(),
            nn.AvgPool2d(3, stride=2, padding=1),
            nn.Flatten(),
            nn.Linear(8192, 2048),
            self.model.fc,
            nn.Softmax(dim=1)
        )

        self.layer4 = nn.Sequential(
            self.model.layer4,
            nn.AvgPool2d(3, stride=2, padding=1),
            nn.Flatten(),
            nn.Linear(8192, 2048),
            self.model.fc,
            nn.Softmax(dim=1)
        )

    def forward(self, img, return_loss=False):
        img = img.to(device='cuda')

        if return_loss:
            return self.forward_train(img)
        else:
            return self.forward_test(img)

    def forward_train(self, x: Tensor) -> Tensor:
        
        x = self.layer1(x)
        y1 = self.earlyExit1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        y2 = self.earlyExit2(x)
        y3 = self.layer4(x)

        return [y1, y2, y3]
    

    def forward_test(self, x: Tensor)-> Tensor:

        bs = x.size()[0]
        y = zeros(bs, 10, device=device('cuda'))
        Mask_Pass_On = ones(bsdevice=device('cuda')).bool()
        x = self.layer1(x)

        if self.activated_branches[0]:
            y_exitOne = self.earlyExit1(x)
            
            Mask_exitOne = max(y_exitOne, axis=1)[0] >= 0.75
            Mask_exitOne = Mask_exitOne.reshape(-1, 1)
            
            # If there are further exits we have to sort the bad results out
            if any(self.activated_branches[1:-1]):
                # pdb.set_trace()
                y_exitOne = y_exitOne * Mask_exitOne
            y += y_exitOne    

            # Invert the mask and reshape it
            Mask_Pass_On = (Mask_exitOne < .5).reshape(-1)

            x = mask_down(x, Mask_Pass_On)

        if any(self.activated_branches[1:-1]):
            x = self.layer2(x)
            x = self.layer3(x)
            
            if self.activated_branches[1]:
                y_exitTwo = self.earlyExit2(x)
                y_exitTwo = mask_up(y_exitTwo, Mask_Pass_On)

                Mask_exitTwo = max(y_exitTwo, axis=1)[0] >= 0.75
                Mask_exitTwo = Mask_exitTwo.reshape(-1, 1)
                
                # If there are further exits we have to sort the bad results out
                if (self.activated_branches[-1]):
                    y_exitTwo = y_exitTwo * Mask_exitTwo
                
                y += y_exitTwo    

                Mask_Pass_On = mask_up(Mask_exitTwo, Mask_Pass_On).reshape(-1)

            if self.activated_branches[-1]:
                x = mask_down(x, Mask_Pass_On)
                x = self.layer4(x)
                
                y_full_path = mask_up(x, Mask_Pass_On)
                
                y += y_full_path

        return y

# BACKBONES.register_module()

# class ResNetExitOne(baseBranch):
    
#     """ My version of the branchyNet with a ResNet
#         and one earlyExit.

#     Args:
#         stack_on_top: Whether the model should be built on a existing
#         baseBranch.
#     """

#     def __init__(self, stack_on_top: bool):

#         super(self).__init__()
    
#         # Load the first layer and ExitOne from a pretrained model

#         if stack_on_top:
#             dirname = Path(__file__).parent.parent.parent.parent
#             # path_baseBranch = dirname /   ###################'work_dirs/resnet50cifar10_backbone.pth'
#             if not (path_baseBranch.is_file()):
#                 sys.exit(f"Pretrained basBranch requiered. {path_baseBranch} is no file.")
#             state_dict = load(path_baseBranch)
#             self.layer1 = state_dict["layer1"]
#             self.exitOne = state_dict["exitOne"]
#             self.stack_on_top = stack_on_top

#             for name, param in self.named_parameters():
#                 if "layer1" in name or "exitOne" in name:
#                     param.requires_grad = False
        
#     def forward(self, img, return_loss=False):
#         if return_loss:
#             return self.forward_train(img)
#         else:
#             return self.forward_test(img)

#     def forward_train(self, x):

#         if self.stack_on_top:    

#             # When stacking on top of baseBranch then we only train the full path

#             x = self.layer1(x)
        
#             x = self.layer2(x)

#             x = self.layer3(x)

#             y = self.layer4(x)

#         return y

#         if not self.stack_on_top:
#             raise NotImplementedError("Independent Implementation of ResNetExitOne not available, yet.")
    

#     def forward_test(self, x):

#         x = self.layer1(x)
    
#         y = self.earlyExit1(x)

#         # Decide: Are we already sure about this input or are we not?
#         # If so return early result .
#         # Otherwise return further processed result.

#         Mask_One = max(y, axis=1)[0] >= 0.75
#         Mask_Pass_On = max(y, axis=1)[0] < 0.75
#         Mask_One = Mask_early.reshape(-1, 1) # (B K)
#         Mask_Pass_On = Mask_full.reshape(-1, 1, 1, 1) # (B C H W)

#         y_early = y * Mask_early

#         x = x * Mask_full

#         x = self.layer2(x)
#         x = self.layer3(x)
#         y_full = self.layer4(x)

#         return y_early + y_full

# BACKBONES.register_module()

# class ExitTwo(ResNetExitOne):
#     """ My version of the branchyNet with only a exitTwo.

#     Args:
#         stack_on_top: Whether the model should be built on a existing
#         baseBranch.
#     """

#     def __init__(self, stack_on_top: bool):

#         super(self).__init__(stack_on_top)
    
#         # Load the first layer and ExitOne from a pretrained model

#         if stack_on_top:
#             dirname = Path(__file__).parent.parent.parent.parent
#             # path_ResNetExitOne = dirname /   ###################'work_dirs/resnet50cifar10_backbone.pth'
#             if not (path_ResNetExitOne.is_file()):
#                 sys.exit(f"Pretrained ResNetExitOne requiered. {path_ResNetExitOne} is no file.")
#             state_dict = load(path_ResNetExitOne)

#             self.layer1 = state_dict["layer1"]
#             self.layer2 = state_dict["layer2"]
#             self.layer3 = state_dict["layer3"]
#             self.layer4 = state_dict["layer4"]
#             self.exitOne = state_dict["exitOne"]
#             self.stack_on_top = stack_on_top
        
#             if self.stack_on_top:    
#                 "If stacking on top we only train exit 2"
#                 for name, param in self.named_parameters():
#                     # Train only exitTwo
#                     freeze_list = ["exitOne", "layer1", "layer2", "layer3"]
#                     if any([layer in name for layer in freeze_list]):
#                         param.requires_grad = False
#             if not self.stack_on_top:
#                 raise NotImplementedError("Independet ExitTwo not implemented, yet.")
            
#     def forward(self, img, return_loss=False):
#         if return_loss:
#             return self.forward_train(img)
#         else:
#             return self.forward_test(img)

#     def forward_train(self, x):

#         x = self.layer1(x)

#         x = self.layer2(x)

#         x = self.layer3(x)

#         y = self.earlyExit2(x)

#         return y

#     def forward_test(self, x):

#         x = self.layer1(x)

#         x = self.layer2(x)

#         x = self.layer3(x)

#         y = self.earlyExit2(x)

# class ResNetExitTwo(ExitTwo):
#     """ My version of the branchyNet with a main path and 
#         one early exit further into the net.

#     Args:
#         stack_on_top: Whether the model should be built on a existing
#         baseBranch.
#     """

#     def __init__(self, stack_on_top: bool):

#         super(self).__init__(stack_on_top)
    
#         # Load the first layer and ExitOne from a pretrained model

#         if stack_on_top:
#             dirname = Path(__file__).parent.parent.parent.parent
#             # path_ResNetExitOne = dirname /   ###################'work_dirs/resnet50cifar10_backbone.pth'
#             if not (path_ExitTwo.is_file()):
#                 sys.exit(f"Pretrained ResNetExitOne requiered. {path_ExitTwo} is no file.")
#             state_dict = load(path_ExitTwo)

#             self.layer1 = state_dict["layer1"]
#             self.layer2 = state_dict["layer2"]
#             self.layer3 = state_dict["layer3"]
#             self.layer4 = state_dict["layer4"]
#             self.exitOne = state_dict["exitOne"]
#             self.exitTwo = state_dict["exitTwo"]
#             self.stack_on_top = stack_on_top
        
#             if self.stack_on_top:    
#                 "If stacking on top we only train exit 2"
#                 for name, param in self.named_parameters():
#                     # Train only Layer 4
#                     freeze_list = ["exitOne", "exitTwo", "layer1", "layer2", "layer3"]
#                     if any([layer in name for layer in freeze_list]):
#                         param.requires_grad = False
#             if not self.stack_on_top:
#                 raise NotImplementedError("Independent ResNet with ExitTwo not implemented, yet.")
            
#     def forward(self, img, return_loss=False):
#         if return_loss:
#             return self.forward_train(img)
#         else:
#             return self.forward_test(img)

#     def forward_train(self, x):

#         x = self.layer1(x)

#         x = self.layer2(x)

#         x = self.layer3(x)

#         y = self.layer4(x)

#         return y

#     def forward_test(self, x):

#         x = self.layer1(x)

#         x = self.layer2(x)

#         x = self.layer3(x)

#         y = self.earlyExit2(x)

def mask_down(t: Tensor, mask: Tensor) -> Tensor:
    return t[mask.bool()]

def mask_up(t: Tensor, mask: Tensor) -> Tensor:

    '''This method takes a downsized vector and upsizes it again, so that the new tensor
        has its values where the mask has its Ones.'''

    mask_as_list = list(mask)
    BS =  len(mask_as_list)
    # BS, C, H, W = len(mask_as_list), *(list(t.size())[1: ])
    small_batch_size = t.size()[0]
    # output = zeros(BS, C, H, W)
    output = zeros(BS, *(list(t.size())[1: ]))

    i = 0
    for j in range(BS):
        if mask_as_list[j]:
            output[j, :] = t[i, :]
            i += 1

    return output

    
    

