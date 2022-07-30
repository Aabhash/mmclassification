import torch
import torch.nn as nn
import torch.nn.functional as F
from .resnet_cifar import ResNet_CIFAR
from ..builder import BACKBONES


class CGConv2d(nn.Conv2d):
    def __init__(
            self,
            in_channels,
            out_channels,
            kernel_size,
            stride=1,
            padding=0,
            dilation=1,
            groups=1,
            bias=False,
            padding_mode='zeros',
            p=4,
            th=-6.0,
            alpha=2.0):
        super(CGConv2d, self).__init__(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            dilation,
            groups,
            bias,
            padding_mode)

        self.partition = p
        self.th = th
        self.alpha = alpha
        self.bn = nn.BatchNorm2d(out_channels, affine=False)

        in_chunk_size = int(in_channels/self.partition)

        mask = torch.zeros(out_channels, in_channels, kernel_size, kernel_size)

        mask[:, 0:in_chunk_size] = torch.ones(
            out_channels, in_chunk_size, kernel_size, kernel_size)

        self.mask = nn.Parameter(mask, requires_grad=False)

        self.threshold = nn.Parameter(
            self.th * torch.ones(1, out_channels, 1, 1))

        self.count_exit = 0
        self.count_all = 0

    def gt(self, input):
        return torch.Tensor.float(torch.gt(input, torch.zeros_like(input)))

    def forward(self, input):
        Yp = F.conv2d(input, self.weight * self.mask, self.bias,
                      self.stride, self.padding, self.dilation, self.groups)

        diff = self.bn(Yp) - self.threshold
        d = self.gt(torch.sigmoid(self.alpha * diff) - 0.5 * torch.ones_like(Yp))

        Y = F.conv2d(input, self.weight, self.bias, self.stride,
                     self.padding, self.dilation, self.groups)

        self.count_all = d.numel()
        self.count_exit = d[d > 0].numel()

        return Y * d + Yp * (torch.ones_like(d) - d)


@BACKBONES.register_module()
class CGResNet(ResNet_CIFAR):
    def __init__(
            self, depth, num_blocks, in_channels=3, base_channels=16, gtarget=1.0, strides=(1, 2, 2), **kwargs):
        super(CGResNet, self).__init__(
            depth,
            in_channels=in_channels,
            stem_channels=base_channels,
            base_channels=base_channels,
            num_stages=len(num_blocks),
            strides=strides,
            dilations=(1, 1, 1),
            out_indices=(2, )
        )
        self.block_expansion = 1
        self.in_planes = base_channels
        self.gtarget = gtarget
        self.conv1 = nn.Conv2d(
            3, self.in_planes, kernel_size=3, stride=1, padding=1, bias=False)
        layers = [self.layer1, self.layer2, self.layer3]
        for l, layer in enumerate(layers):
            pl = 2**(l+4)
            for i, blk in enumerate(layer):
                stride = strides[l] if i == 0 else 1
                blk.conv1 = CGConv2d(
                    self.in_planes,
                    pl,
                    kernel_size=3,
                    stride=stride,
                    padding=1,
                    bias=False,
                    p=kwargs['partitions'],
                    th=kwargs['ginit'],
                    alpha=kwargs['alpha'],
                    # use_group=kwargs['use_group'],
                    # shuffle=kwargs['shuffle'],
                    # sparse_bp=kwargs['sparse_bp']
                )
                self.in_planes = pl * self.block_expansion

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = F.avg_pool2d(out, out.size()[3])
        out = out.view(out.size(0), -1)
        return out
