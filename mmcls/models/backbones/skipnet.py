""" This file contains the model definitions for both original ResNet (6n+2
layers) and SkipNets.
"""

import torch
import torch.nn as nn
import math
from torch.autograd import Variable
import torch.autograd as autograd
from ..builder import BACKBONES
from .base_backbone import BaseBackbone, BaseModule


class BasicBlock(BaseModule):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)
        return out


def conv3x3(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)
# Feedforward-Gate (FFGate-I)
class FeedforwardGateI(BaseModule):
    """ Use Max Pooling First and then apply to multiple 2 conv layers.
    The first conv has stride = 1 and second has stride = 2"""
    def __init__(self, pool_size=5, channel=10):
        super(FeedforwardGateI, self).__init__()
        self.pool_size = pool_size
        self.channel = channel

        self.maxpool = nn.MaxPool2d(4)
        self.conv1 = conv3x3(channel, channel)
        self.bn1 = nn.BatchNorm2d(channel)
        self.relu1 = nn.ReLU(inplace=True)

        # adding another conv layer
        self.conv2 = conv3x3(channel, channel, stride=2)
        self.bn2 = nn.BatchNorm2d(channel)
        self.relu2 = nn.ReLU(inplace=True)

        pool_size = math.floor(pool_size/2)  # for max pooling
        pool_size = math.floor(pool_size/2 + 0.5)  # for conv stride = 2

        self.avg_layer = nn.AvgPool2d(pool_size)
        self.linear_layer = nn.Conv2d(in_channels=channel, out_channels=2,
                                      kernel_size=1, stride=1)
        self.prob_layer = nn.Softmax()
        self.logprob = nn.LogSoftmax()

    def forward(self, x):
        x = self.maxpool(x)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)

        x = self.avg_layer(x)
        x = self.linear_layer(x).squeeze()
        softmax = self.prob_layer(x)
        logprob = self.logprob(x)

        # discretize output in forward pass.
        # use softmax gradients in backward pass
        x = (softmax[:, 1] > 0.5).float().detach() - \
            softmax[:, 1].detach() + softmax[:, 1]

        x = x.view(x.size(0), 1, 1, 1)
        return x, logprob


# soft gate v3 (matching FFGate-I)
class SoftGateI(BaseModule):
    """This module has the same structure as FFGate-I.
    In training, adopt continuous gate output. In inference phase,
    use discrete gate outputs"""
    def __init__(self, pool_size=5, channel=10):
        super(SoftGateI, self).__init__()
        self.pool_size = pool_size
        self.channel = channel

        self.maxpool = nn.MaxPool2d(4)
        self.conv1 = conv3x3(channel, channel)
        self.bn1 = nn.BatchNorm2d(channel)
        self.relu1 = nn.ReLU(inplace=True)

        # adding another conv layer
        self.conv2 = conv3x3(channel, channel, stride=2)
        self.bn2 = nn.BatchNorm2d(channel)
        self.relu2 = nn.ReLU(inplace=True)

        pool_size = math.floor(pool_size/2)  # for max pooling
        pool_size = math.floor(pool_size/2 + 0.5)  # for conv stride = 2

        self.avg_layer = nn.AvgPool2d(pool_size)
        self.linear_layer = nn.Conv2d(in_channels=channel, out_channels=2,
                                      kernel_size=1, stride=1)
        self.prob_layer = nn.Softmax()
        self.logprob = nn.LogSoftmax()

    def forward(self, x):
        x = self.maxpool(x)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)

        x = self.avg_layer(x)
        x = self.linear_layer(x).squeeze()
        softmax = self.prob_layer(x)
        logprob = self.logprob(x)

        x = softmax[:, 1].contiguous()
        x = x.view(x.size(0), 1, 1, 1)

        if not self.training:
            x = (x > 0.5).float()
        return x, logprob


# FFGate-II
class FeedforwardGateII(BaseModule):
    """ use single conv (stride=2) layer only"""
    def __init__(self, pool_size=5, channel=10):
        super(FeedforwardGateII, self).__init__()
        self.pool_size = pool_size
        self.channel = channel

        self.conv1 = conv3x3(channel, channel, stride=2)
        self.bn1 = nn.BatchNorm2d(channel)
        self.relu1 = nn.ReLU(inplace=True)

        pool_size = math.floor(pool_size/2 + 0.5) # for conv stride = 2

        self.avg_layer = nn.AvgPool2d(pool_size)
        self.linear_layer = nn.Conv2d(in_channels=channel, out_channels=2,
                                      kernel_size=1, stride=1)
        self.prob_layer = nn.Softmax()
        self.logprob = nn.LogSoftmax()

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)

        x = self.avg_layer(x)
        x = self.linear_layer(x).squeeze()
        softmax = self.prob_layer(x)
        logprob = self.logprob(x)

        # discretize
        x = (softmax[:, 1] > 0.5).float().detach() - \
            softmax[:, 1].detach() + softmax[:, 1]

        x = x.view(x.size(0), 1, 1, 1)
        return x, logprob




@BACKBONES.register_module()
class ResNetFeedForwardSP(BaseBackbone):
    """ SkipNets with Feed-forward Gates for Supervised Pre-training stage.
    Adding one routing module after each basic block."""

    def __init__(self, layers, num_classes=10,
                 gate_type='fisher', **kwargs):
        self.inplanes = 64
        super(ResNetFeedForwardSP, self).__init__()

        self.num_layers = layers
        self.conv1 =  self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        block = BasicBlock
        self.gate_type = gate_type
        self._make_group(block, 64, layers[0],gate_type=gate_type, group_id=1, pool_size=56)
        self._make_group(block, 128, layers[1],gate_type=gate_type, group_id=2, pool_size=28)
        self._make_group(block, 256, layers[2], gate_type = gate_type, group_id=3, pool_size=14)
        self._make_group(block, 512, layers[3],  gate_type = gate_type, group_id=4, pool_size=7)
        self.avgpool = nn.AvgPool2d(7)
        
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                n = m.weight.size(0) * m.weight.size(1)
                m.weight.data.normal_(0, math.sqrt(2. / n))

    def _make_group(self, block, planes, layers, group_id=1,
                    gate_type='fisher', pool_size=16):
        """ Create the whole group"""
        for i in range(layers):
            if group_id > 1 and i == 0:
                stride = 2
            else:
                stride = 1

            meta = self._make_layer_v2(block, planes, stride=stride,
                                       gate_type=gate_type,
                                       pool_size=pool_size)

            setattr(self, 'group{}_ds{}'.format(group_id, i), meta[0])
            setattr(self, 'group{}_layer{}'.format(group_id, i), meta[1])
            setattr(self, 'group{}_gate{}'.format(group_id, i), meta[2])

    def _make_layer_v2(self, block, planes, stride=1,
                       gate_type='fisher', pool_size=16):
        """ create one block and optional a gate module """
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),

            )
        layer = block(self.inplanes, planes, stride, downsample)
        self.inplanes = planes * block.expansion

        if gate_type == 'ffgate1':
            gate_layer = FeedforwardGateI(pool_size=pool_size,
                                          channel=planes*block.expansion)
        elif gate_type == 'ffgate2':
            gate_layer = FeedforwardGateII(pool_size=pool_size,
                                           channel=planes*block.expansion)
        else:
            gate_layer = None

        if downsample:
            return downsample, layer, gate_layer
        else:
            return None, layer, gate_layer

    def forward(self, x):
        """Return output logits, masks(gate ouputs) and probabilities
        associated to each gate."""

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        self.masks = []
        gprobs = []
        # must pass through the first layer in first group
        x = getattr(self, 'group1_layer0')(x)
        # gate takes the output of the current layer

        mask, gprob = getattr(self, 'group1_gate0')(x)
        gprobs.append(gprob)
        self.masks.append(mask.squeeze())
        prev = x  # input of next layer
        #going through the blocks
        for g in range(3):
            #going through the layers
            for i in range(0 + int(g == 0), self.num_layers[g]):
                if getattr(self, 'group{}_ds{}'.format(g+1, i)) is not None:
                    prev = getattr(self, 'group{}_ds{}'.format(g+1, i))(prev)
                x = getattr(self, 'group{}_layer{}'.format(g+1, i))(x)
                prev = x = mask.expand_as(x) * x \
                           + (1 - mask).expand_as(prev) * prev
                mask, gprob = getattr(self, 'group{}_gate{}'.format(g+1, i))(x)
                gprobs.append(gprob)
                self.masks.append(mask.squeeze())

        del self.masks[-1]

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        #x = self.fc(x)
        return x 
        return x, masks, gprobs

