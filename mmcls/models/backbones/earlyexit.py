# Copyright (c) Carl. All rights reserved.
from xmlrpc.client import Boolean, boolean
from torch import Tensor, zeros, ones, device, cuda, logical_and, logical_not
import torch.nn as nn
from torch.hub import load_state_dict_from_url
# from torchvision.models.resnet import ResNet, Bottleneck
from torch import load, save, sum, max # WIP: Not needed in final version I guess
from pathlib import Path
from collections import OrderedDict

import sys
import pdb

from ..builder import BACKBONES
from .base_backbone import BaseBackbone
from . import ResNet_CIFAR

@BACKBONES.register_module()
class BranchyNet(nn.Module):
    
    """ The BaseBlock of my BranchyNet Version. 
        It contains the first layer and the first exit.
        It also keeps hold of the whole loaded ResNet.

    """

    def __init__(self, activated_branches: list, pretrained: Boolean=False):

        super(BranchyNet, self).__init__()

        # The variable activated branches stores which Branches to use during Test Phase
        assert(any(activated_branches))
        self.activated_branches = activated_branches.copy()

        self.resnet = ResNet_CIFAR(depth=50)
        self.fc = nn.Linear(2048, 10, bias=True)

        if pretrained:
        # Load Pretrained Resnet 
            dirname = Path(__file__).parent.parent.parent.parent
            resnet_path_backbone = dirname /  'work_dirs/resnet50cifar10_backbone.pth'
            if not (resnet_path_backbone.is_file()):
                sys.exit(f"class exitOne requieres pretrained resNet. {resnet_path_backbone} is no file.")
            
            state_dict = load(resnet_path_backbone)
            state_dict = OrderedDict([(k.replace("backbone.", "").replace("head.", ""),v) for k,v in state_dict.items()])

            self.resnet.load_state_dict(state_dict)

        # self.layer1 = self.resnet.layer1

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
            self.fc,
            nn.Softmax(dim=1)
        )

        self.layer2 = self.resnet.layer2

        self.layer3 = self.resnet.layer3

        self.earlyExit2 = nn.Sequential(
            nn.Conv2d(1024, 2048, 3, 2),
            nn.BatchNorm2d(2048, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.ReLU(),
            nn.AvgPool2d(3, stride=2, padding=1),
            nn.Flatten(),
            nn.Linear(8192, 2048),
            self.fc,
            nn.Softmax(dim=1)
        )

        self.layer4 = self.resnet.layer4

        self.layer4_head = nn.Sequential(    
           nn.AvgPool2d(3, stride=2, padding=1),
           nn.Flatten(),
           nn.Linear(8192, 2048),
           self.fc,
           nn.Softmax(dim=1)
        )

        # self.head_1 = nn.AvgPool2d(3, stride=2, padding=1)
        # self.head_2 = nn.Flatten()
        # self.head_3 = nn.Linear(8192, 2048)
        # self.head_4 = self.resnet.fc
        # self.head_5 = nn.Softmax(dim=1)


    def forward(self, img, return_loss=False):
        if cuda.is_available():
            img = img.to(device='cuda')

        if return_loss:
            return self.forward_train(img)
        else:
            return self.forward_test(img)

    def forward_train(self, x: Tensor) -> Tensor:
        # x = self.preprocess(x)
        x = self.resnet.layer1(x)
        y1 = self.earlyExit1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        y2 = self.earlyExit2(x)
        x = self.layer4(x)
        y3 = self.layer4_head(x)

        return [y1, y2, y3]

    def simple_test(self, x: Tensor) -> Tensor:
        return forward_test(self, x)
    
    def forward_test(self, x: Tensor)-> Tensor:
        
        pdb.set_trace()

        bs = x.size()[0]
        y = zeros(bs, 10)
        Mask_Pass_On = ones(bs).bool()

        if cuda.is_available():
            y = y.to(device='cuda')
            Mask_Pass_On.to(device='cuda')

        x = self.resnet.layer1(x)
        
        if self.activated_branches[0]:
            y_exitOne = self.earlyExit1(x)
            
            Mask_exitOne = max(y_exitOne, axis=1)[0] >= 0.65
            Mask_exitOne = Mask_exitOne.reshape(-1, 1)
            
            # If there are further exits we have to sort the bad results out
            if any(self.activated_branches[1:-1]):
                y_exitOne = y_exitOne * Mask_exitOne
            y += y_exitOne    

            # Invert the mask and reshape it
            Mask_Pass_On = (Mask_exitOne < .5).reshape(-1)


        if any(self.activated_branches[1:]):
            x = mask_down(x, Mask_Pass_On)
            
            x = self.layer2(x)
            x = self.layer3(x)
            
            if self.activated_branches[1]:

                y_exitTwo = self.earlyExit2(x)
                y_exitTwo = mask_up(y_exitTwo, Mask_Pass_On)
                x = mask_up(x, Mask_Pass_On)
                

                Mask_exitTwo = max(y_exitTwo, axis=1)[0] >= 0.65
                Mask_exitTwo = Mask_exitTwo.reshape(-1, 1)
                
                # If there are further exits we have to sort the bad results out
                if (self.activated_branches[-1]):
                    y_exitTwo = (y_exitTwo * Mask_exitTwo)
                
                    if cuda.is_available():
                        y_exitTwo = y_exitTwo.to(device='cuda')
                
                y += y_exitTwo    
                
                Mask_Pass_On = logical_not(Mask_exitTwo).reshape(-1)

                x = mask_down(x, Mask_Pass_On)

            if self.activated_branches[-1]:
                x = self.layer4(x)
                x = self.layer4_head(x)

                y_full_path = mask_up(x, Mask_Pass_On)
                
                y += y_full_path

        return y


@BACKBONES.register_module()
class BranchyNetImagenette(nn.Module):
    
    """ The BaseBlock of my BranchyNet Version. 
        It contains the first layer and the first exit.
        It also keeps hold of the whole loaded ResNet.

    """

    def __init__(self, activated_branches: list, pretrained: boolean=False):

        super(BranchyNetImagenette, self).__init__()

        # The variable activated branches stores which Branches to use during Test Phase
        assert(any(activated_branches))
        self.activated_branches = activated_branches.copy()
        self.pretrained = pretrained

        self.model = ResNet_CIFAR(depth=50)

        # Load Pretrained Resnet 
        if self.pretrained:
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
            self.model.layer1,
            nn.Conv2d(256, 256, 5, 3),
            nn.BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.ReLU(),

        )

        self.earlyExit1 = nn.Sequential(
            nn.Conv2d(256, 512, 5, 3),
            nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.ReLU(),
            nn.AvgPool2d(5, stride=3, padding=0),
            nn.Conv2d(512, 1024, 3, 2),
            nn.BatchNorm2d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.ReLU(),
            nn.AvgPool2d(3, stride=2, padding=1),
            nn.Flatten(),
            nn.Linear(4096, 10),
            nn.Softmax(dim=1),
        )

        self.layer2 = self.model.layer2

        self.earlyExit2 = nn.Sequential(
            nn.Conv2d(512, 512, 3, 3),
            nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.ReLU(), 
            nn.Conv2d(512, 1024, 3, 2),
            nn.BatchNorm2d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.ReLU(),
            nn.AvgPool2d(3, stride=2, padding=0),
            nn.Flatten(),
            nn.Linear(4096, 10),
            nn.Softmax(dim=1),
        )

        self.layer3 = self.model.layer3

        self.layer4 = nn.Sequential(
            self.model.layer4,
            nn.Conv2d(2048, 1024, 3, 2, padding=0),
            nn.BatchNorm2d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.ReLU(),
            nn.AvgPool2d(2, stride=2, padding=0),
            nn.Flatten(),
            nn.Linear(4096, 10),
            nn.Softmax(dim=1)
        )

    def forward(self, img, return_loss=False):
        if cuda.is_available():
            img = img.to(device='cuda')

        if return_loss:
            return self.forward_train(img)
        else:
            return self.forward_test(img)

    def forward_train(self, x: Tensor) -> Tensor:
        x = self.layer1(x)
        y1 = self.earlyExit1(x)
        x = self.layer2(x)
        y2 = self.earlyExit2(x)
        x = self.layer3(x)
        y3 = self.layer4(x)

        return [y1, y2, y3]

    def simple_test(self, x: Tensor) -> Tensor:
        return forward_test(self, x)
    
    def forward_test(self, x: Tensor)-> Tensor:

        bs = x.size()[0]
        y = zeros(bs, 10)
        Mask_Pass_On = ones(bs).bool()

        if cuda.is_available():
            y = y.to(device='cuda')
            Mask_Pass_On.to(device='cuda')

        x = self.layer1(x)

        # 1 2 3 4 5 6 7 

        if self.activated_branches[0]:
            pdb.set_trace()
            y_exitOne = self.earlyExit1(x)
            
            Mask_exitOne = max(y_exitOne, axis=1)[0] >= 0.65
            Mask_exitOne = Mask_exitOne.reshape(-1, 1)
            
            # If there are further exits we have to sort the bad results out
            if any(self.activated_branches[1:-1]):
                # pdb.set_trace()
                y_exitOne = y_exitOne * Mask_exitOne
            y += y_exitOne    

            # Invert the mask and reshape it
            Mask_Pass_On = (Mask_exitOne < .5).reshape(-1)

        if any(self.activated_branches[1:]):
            x = mask_down(x, Mask_Pass_On)       
            x = self.layer2(x)
            if self.activated_branches[1]:
                pdb.set_trace()
                y_exitTwo = self.earlyExit2(x)
                y_exitTwo = mask_up(y_exitTwo, Mask_Pass_On)
                x = mask_up(x, Mask_Pass_On)

                Mask_exitTwo = max(y_exitTwo, axis=1)[0] >= 0.65
                Mask_exitTwo = Mask_exitTwo.reshape(-1, 1)
                
                # If there are further exits we have to sort the bad results out
                if (self.activated_branches[-1]):
                    y_exitTwo = (y_exitTwo * Mask_exitTwo)
                    # x = mask_up(x, Mask_exitTwo)
                    if cuda.is_available():
                        y_exitTwo = y_exitTwo.to(device='cuda')
                    # print(y.get_device(), y_exitTwo.get_device())
                y += y_exitTwo    
                
                Mask_Pass_On = logical_not(Mask_exitTwo).reshape(-1)

                x = mask_down(x, Mask_Pass_On)
                # Mask_Pass_On -= mask_up(Mask_exitTwo, Mask_Pass_On).reshape(-1)

            if self.activated_branches[-1]:
                x = self.layer3(x)
                x = self.layer4(x)
                
                y_full_path = mask_up(x, Mask_Pass_On)
                
                y += y_full_path

        return y

@BACKBONES.register_module()
class BranchyNetImagenette2(nn.Module):
    
    """ The BaseBlock of my BranchyNet Version. 
        It contains the first layer and the first exit.
        It also keeps hold of the whole loaded ResNet.

    """

    def __init__(self, activated_branches: list, pretrained: boolean=False,
                       exit_treshholds: list=[0.6, 0.6]):
        """activated_branches: list of which branches to use. 
                                [True, True, True] means use all
           pretrained: Use pretrained ResNet for the main path of the NN
           exit_treshholds: By which probability should the early
                            exits be left  
            """


        super(BranchyNetImagenette2, self).__init__()

        # The variable activated branches stores which Branches to use during Test Phase
        assert(any(activated_branches))
        self.activated_branches = activated_branches.copy()
        self.pretrained = pretrained
        
        if self.activated_branches[0] == True:
            self.th_One = exit_treshholds.pop(0)
        if self.activated_branches[1] == True:
            self.th_Two = exit_treshholds.pop(0)

        if cuda.is_available():
            self.device = 'cuda'
        else:
            self.device = 'cpu'            

        self.model = ResNet_CIFAR(depth=50).to(self.device)
        # self.model.fc = nn.Linear(2048, 10, bias=True)

        # Load Pretrained Resnet 
        if self.pretrained:
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
            self.model.layer1,
            nn.Conv2d(256, 256, 5, 3),
            nn.BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.ReLU(),
        ).to(self.device)

        self.earlyExit1 = nn.Sequential(
            nn.Conv2d(256, 256, 5, 4),
            nn.BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.ReLU(),
            nn.Conv2d(256, 256, 3, 2),
            nn.BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.ReLU(),
            nn.Conv2d(256, 256, 3, 2),
            nn.BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.ReLU(),
            nn.AvgPool2d(2, stride=1, padding=1),
            nn.Flatten(),
            nn.Linear(4096, 10),
            nn.Softmax(dim=1),
        ).to(self.device)

        self.layer2 = self.model.layer2.to(self.device)

        self.earlyExit2 = nn.Sequential(
            nn.Conv2d(512, 512, 5, 4),
            nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.ReLU(),
            nn.Conv2d(512, 512, 3, 3),
            nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.ReLU(), 
            nn.Conv2d(512, 512, 3, 2),
            nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.ReLU(),
            nn.AvgPool2d(2, stride=1, padding=1),
            nn.Flatten(),
            nn.Linear(2048, 10),
            nn.Softmax(dim=1),
        ).to(self.device)

        self.layer3 = self.model.layer3.to(self.device)

        self.layer4 = nn.Sequential(
            self.model.layer4,
            nn.Conv2d(2048, 512, 5, 2, padding=0),
            nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.ReLU(),
            nn.AvgPool2d(2, stride=1, padding=0),
            nn.Flatten(),
            nn.Linear(2048, 10),
            nn.Softmax(dim=1)
        ).to(self.device)

    def forward(self, img, return_loss=False):
        
        img = img.to(self.device)

        if return_loss:
            return self.forward_train(img)
        else:
            return self.forward_test(img)

    def forward_train(self, x: Tensor) -> Tensor:
        x = x.to(self.device)
        x = self.layer1(x)
        y1 = self.earlyExit1(x)
        x = self.layer2(x)
        y2 = self.earlyExit2(x)
        x = self.layer3(x)
        y3 = self.layer4(x)

        return [y1, y2, y3]

    def simple_test(self, x: Tensor) -> Tensor:
        return forward_test(self, x)
    
    def forward_test(self, x: Tensor)-> Tensor:
        x = x.to(self.device)
        bs = x.size()[0]
        y = zeros(bs, 10).to(self.device)
        Mask_Pass_On = ones(bs).bool()
        Mask_Pass_On.to(self.device)

        x = self.layer1(x)

        if self.activated_branches[0]:
            y_exitOne = self.earlyExit1(x)
            
            Mask_exitOne = max(y_exitOne, axis=1)[0] >= self.th_One
            Mask_exitOne = Mask_exitOne.reshape(-1, 1)
            
            # If there are further exits we have to sort the bad results out
            if any(self.activated_branches[1:-1]):
                # pdb.set_trace()
                y_exitOne = y_exitOne * Mask_exitOne
            y += y_exitOne.to(self.device)    

            # Invert the mask and reshape it
            Mask_Pass_On = (Mask_exitOne < .5).to(self.device)

        if any(self.activated_branches[1:]):
            x = mask_down(x, Mask_Pass_On).to(self.device)       
            x = self.layer2(x)
            if self.activated_branches[1]:

                y_exitTwo = self.earlyExit2(x)
                y_exitTwo = mask_up(y_exitTwo, Mask_Pass_On).to(self.device)
                x = mask_up(x, Mask_Pass_On).to(self.device)

                Mask_exitTwo = max(y_exitTwo, axis=1)[0] >= self.th_Two
                Mask_exitTwo = Mask_exitTwo.to(self.device)
                
                # If there are further exits we have to sort the bad results out
                if (self.activated_branches[-1]):
                    y_exitTwo = (Mask_exitTwo.reshape(-1,1)*y_exitTwo)
                    y_exitTwo = y_exitTwo.to(self.device)
                y += y_exitTwo    
                
                Mask_Pass_On = logical_not(Mask_exitTwo).to(self.device)

                x = mask_down(x, Mask_Pass_On).to(self.device)

            if self.activated_branches[-1]:
                x = self.layer3(x)
                x = self.layer4(x)
                
                y_full_path = mask_up(x, Mask_Pass_On).to(self.device)
                
                y += y_full_path

        return y


def mask_down(t: Tensor, mask: Tensor) -> Tensor:
    
    mask = mask.reshape(-1)
    if cuda.is_available():
        mask=mask.to('cuda')

    return t[mask.bool()]

def mask_up(t: Tensor, mask: Tensor) -> Tensor:
    '''This method takes a downsized vector and upsizes it again, so that the new tensor
        has its values where the mask has its Ones.'''
    mask = mask.reshape(-1)
    if cuda.is_available():
        mask=mask.to('cuda')
        t=t.to('cuda')

    bs = mask.size()[0]
    output = zeros(bs, *(list(t.size())[1: ]))

    if cuda.is_available():
        output=output.to('cuda')
    output[mask] = t

    return output
    

