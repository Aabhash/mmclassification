# Copyright (c) Carl. All rights reserved.
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
from ..backbones import ResNet_CIFAR

@BACKBONES.register_module()

class res_inter_classifiers(nn.Module):
    # ToDo: Add input information
    
    """`A resnet with intermediate classifiers backbone.

    Args:
        path_resnet: The path where the base resnet can be found
    """

    def __init__(self, path_resnet: str = None):

        super(res_inter_classifiers, self).__init__()

        print(f"Path_Resnet: {path_resnet}")

        # super(ResNet, self).__init__()

        self.model = ResNet_CIFAR(depth=50)
        # self.model2 = ResNet(block=Bottleneck, layers = [3, 4, 6, 3], num_classes=10)

        # ResNet will be pretrained in the project group at one point
        # Alternative: Get ResNet from the web

        if path_resnet == None:

            dirname = Path(__file__).parent.parent.parent.parent
            print(dirname)
            resnet_path_backbone = dirname /  'work_dirs/resnet50cifar10_backbone.pth'
            # model_path = dirname /  'results_early_exit/checkpoints/EE-resnet50-pytorch.pth'

            print(f"resnet_path_backbone: {resnet_path_backbone}")

            if (resnet_path_backbone).is_file():
                
                state_dict = load(resnet_path_backbone)
            
                state_dict = OrderedDict([(k.replace("backbone.", "").replace("head.", ""),v) for k,v in state_dict.items()])
                

                self.model.fc = nn.Linear(2048, 10, bias=True)
                self.model.load_state_dict(state_dict)

                print(f"{resnet_path_backbone} used.")

            save(self.model.state_dict(), resnet_path_backbone)

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

            # self.early_exit_2 = 

            self.layer3 = self.model.layer3

            # self.early_exit_3 = 

            self.layer4 = nn.Sequential(
                self.model.layer4,
                nn.AvgPool2d(3, stride=2, padding=1),
                nn.Flatten(),
                nn.Linear(8192, 2048),
                self.model.fc,
                nn.Softmax(dim=1)
            )

            # if (model_path).is_file():
            #     state_dict = load(model_path)
            #     self.model.load_state_dict(state_dict)
            #     print(f"{model_path} used.")

        # self.num_classes = num_classes   
        
    def forward(self, img, return_loss=False):
        if return_loss:
            return self.forward_train(img)
        else:
            return self.forward_test(img)

    def forward_train(self, x):
        
        x = self.layer1(x)
        
        y1 = self.earlyExit1(x)

        # [1, 256, H/4, W/4]
        x = self.layer2(x)
        # [1, 512,  H/8, W/8]
        x = self.layer3(x)
        # [1, 1024, H/16, W/16]

        y2 = self.layer4(x)
        # [1, 10]
        return y1
        return 0.5 * y1 + 0.5 * y2

    def forward_test(self, x):

            x = self.layer1(x)

            y1 = self.earlyExit1(x)

            if max(y1) > 0.8:
                return y1

            # [1, 256, H/4, W/4]
            x = self.layer2(x)
            # [1, 512,  H/8, W/8]
            x = self.layer3(x)
            # [1, 1024, H/16, W/16]

            y2 = self.layer4(x)
            # [1, 10]
            return y1


    # def forward(self, x):

    #     x = self.layer1(x)

    #     y1 = self.earlyExit1(x)

    #     # [BS, 256, H/4, W/4]
    #     x = self.layer2(x)
    #     # [BS, 512,  H/8, W/8]
    #     x = self.layer3(x)
    #     # [BS, 1024, H/16, W/16]

    #     y2 = self.layer4(x)
    #     # [BS, 10]

    #     return 0.5 * y1 + 0.5 * y2


"""
# if __name__=="__main__":
#     m = res_inter_classifiers()
    
#     t = rand(1, 3, 64, 64)

#     # dirname = Path(__file__).parent.parent.parent
#     # model_path = dirname /  'results_early_exit/checkpoints/EE-resnet50-pytorch.pth'

#     # save(m.model.state_dict(), model_path)
#     m.forward_train(t)"""