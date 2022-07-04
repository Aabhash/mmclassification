from lib2to3.pytree import Base
import torch
import torch.nn as nn
import torch.nn.functional as F
import copy

from ..builder import BACKBONES
from .base_backbone import BaseBackbone


class InitialLayer(nn.Module):
    """
    First layer of RANet, Generate base features.
    """
    def __init__(self, in_c, out_c, n_scales=3):
        super().__init__()
        self.layers = nn.ModuleList()

        # self.layers.append(nn.Sequential(
        #     nn.Conv2d(
        #         in_c,
        #         out_c,
        #         kernel_size=3, 
        #         stride=1,
        #         padding=1, 
        #         bias=False
        #     ),
        #     nn.BatchNorm2d(out_c),
        #     nn.ReLU(inplace=True)
        # ))

        # in_c = copy.copy(out_c)

        for i in range(n_scales):
            stride = 1 if i == 0 else 2
            out_channel = out_c * (2**i)
            self.layers.append(nn.Sequential(
                nn.Conv2d(
                    in_c,
                    out_channel,  # 1, 2, 4, 8...
                    kernel_size=3,
                    stride=stride,
                    padding=1, 
                    bias=False
                ),
                nn.BatchNorm2d(out_channel),
                nn.ReLU(inplace=True)
            ))
            in_c = copy.copy(out_channel)

    def forward(self, x):
        """
        Forward across all scales
        """
        ret = []
        for layer in self.layers:
            x = layer(x)
            ret.append(x)
        return ret[::-1]        


class ConvBlock(nn.Module):
    def __init__(self, in_c, out_c, width, type="same"):
        super().__init__()

        mid_c = min(in_c, width * out_c)

        if type == "down":
            end_stride = 2
        else:
            end_stride = 1

        self.net = nn.Sequential(
            nn.Conv2d(
                in_c,
                mid_c,
                kernel_size=1,
                stride=1,
                padding=0,
                bias=False
            ),
            nn.BatchNorm2d(mid_c),
            nn.ReLU(True),
            nn.Conv2d(
                mid_c,
                out_c,
                kernel_size=3,
                stride=end_stride,
                padding=1,
                bias=False
            ),
            nn.BatchNorm2d(out_c),
            nn.ReLU(True),
        )

    def forward(self, x):
        if not isinstance(x, list):
            x = [x]
        return torch.cat([x[0], self.net(x[0])], dim=1)


class ConvUpSampleBlock(nn.Module):
    def __init__(self, in_c, in_c2, out_c, compress_factor, width, width2, down_type="same"):
        super().__init__()
        layer = []
        reduce_c = int(torch.floor(torch.tensor(out_c * compress_factor)))
        end_c = out_c - reduce_c
        mid_c = min(in_c, width * end_c)

        if down_type == "same":
            end_stride = 1
        else:
            end_stride = 2

        self.net = nn.Sequential(
            # nn.BatchNorm2d(in_c),
            # nn.ReLU(True),
            nn.Conv2d(
                in_c,
                mid_c,
                kernel_size=1,
                stride=1,
                padding=0,
                bias=False
            ),
            nn.BatchNorm2d(mid_c),
            nn.ReLU(True),
            nn.Conv2d(
                mid_c,
                end_c,
                kernel_size=3,
                stride=end_stride,
                padding=1,
                bias=False
            ),
            nn.BatchNorm2d(end_c),
            nn.ReLU(True)
        )
        mid_c2 = min(in_c2, width2 * reduce_c)

        self.up_net = nn.Sequential(
            # nn.BatchNorm2d(in_c2),
            # nn.ReLU(True),
            nn.Conv2d(
                in_c2,
                mid_c2,
                kernel_size=1,
                stride=1,
                padding=0,
                bias=False
            ),
            nn.BatchNorm2d(mid_c2),
            nn.ReLU(True),
            nn.Conv2d(
                mid_c2,
                reduce_c,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False
            ),
            nn.BatchNorm2d(reduce_c),
            nn.ReLU(True)
        )

    def forward(self, x):
        out = self.net(x[1])
        shape = out.size()
        out = [
            F.interpolate(
                x[1], 
                size=(shape[2], shape[3]),
                mode='bilinear',
                align_corners=True
            ),
            F.interpolate(
                self.up_net(x[0]),
                size=(shape[2], shape[3]),
                mode="bilinear",
                align_corners=True
            ),
            out
        ]
        return torch.cat(out, dim=1)

class DenseBlock(nn.Module):
    def __init__(self, nlayers, in_c, growth_rate, reduction_rate, transition, bnwidth):
        super().__init__()
        self.layers = nn.ModuleList()
        self.nlayers = nlayers
        self.gr = growth_rate
        self.rr = reduction_rate
        self.bnwidth = bnwidth
        self.in_c = in_c
        self.out_c = in_c + nlayers* growth_rate
        self.transition = transition
        self.type = "basic"
        self._set_up_blocks()

    def _set_up_blocks(self):
        for i in range(self.nlayers):
            self.layers.append(
                ConvBlock(
                    self.in_c + i * self.gr,
                    self.gr,
                    self.bnwidth,
                    type="same"
                )
            )
        if self.transition:
            out_channels = torch.floor(torch.tensor(1 * self.rr * self.out_c))
            self.transition_block = nn.Sequential(
                nn.Conv2d(
                    self.out_c,
                    out_channels,
                    kernel_size=1,
                    stride=1,
                    padding=0
                ),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(True)
            )
    
    def forward(self, x):
        all_output = [x]
        for layer in self.layers:
            x = layer(x)
            all_output.append(x)
        if self.transition:
            return self.transition_block(all_output[-1]), all_output
        else:
            return all_output[-1], all_output


class DenseFusionBlock(DenseBlock):
    def __init__(self, nlayers, in_c, in_c_lf, growth_rate, reduction_rate, transition, compress_factor, bnwidth1, bnwidth2, down_type="same"):
        self.in_c_lf = in_c_lf
        self.compress_factor = compress_factor
        self.bnwidth2 = bnwidth2
        self.down_type = down_type
        super().__init__(nlayers, in_c, growth_rate, reduction_rate, transition, bnwidth1)
        self._set_up_blocks()
        

    def _set_up_blocks(self):
        self.type = "fusion"
        for i in range(self.nlayers):
            self.layers.append(
                ConvUpSampleBlock(
                    self.in_c + i * self.gr,
                    self.in_c_lf[i],
                    self.gr,
                    self.compress_factor,
                    self.bnwidth,
                    self.bnwidth2,
                    down_type="same"
                )
            )
        self.layers.append(
            ConvUpSampleBlock(
                self.in_c + (i+1) * self.gr,
                self.in_c_lf[i+1],
                self.gr,
                self.compress_factor,
                self.bnwidth,
                self.bnwidth2,
                down_type=self.down_type
            )
        )

        self.lastconv = nn.Sequential(
                nn.Conv2d(
                    self.in_c_lf[self.nlayers],
                    int(torch.floor(torch.tensor(self.out_c*self.compress_factor))),
                    kernel_size=1,
                    stride=1,
                    padding=0, 
                    bias=False
                ),
                nn.BatchNorm2d(int(self.out_c*self.compress_factor)),
                nn.ReLU(inplace=True)
            )

        self.out_c += int(torch.floor(torch.tensor(self.out_c * self.compress_factor)))

        if self.transition:
            out_channels = int(torch.floor(torch.tensor(1 * self.rr * self.out_c)))
            self.transition_block = nn.Sequential(
                nn.Conv2d(
                    self.out_c,
                    out_channels,
                    kernel_size=1,
                    stride=1,
                    padding=0
                ),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(True)
            )
    
    def forward(self, x, lf):
        all_output = [x]
        for i in range(self.nlayers):
            x = self.layers[i]([lf[i], x])
            all_output.append(x)
        
        shape = all_output[-1].size()
        output = torch.cat([all_output[-1],
            F.interpolate(
                self.lastconv(lf[self.nlayers]),
                size=(shape[2], shape[3]),
                mode="bilinear",
                align_corners=True
            )
            ], dim=1)
        if self.transition:
            return self.transition_block(output), all_output
        else:
            return output, all_output


class Classifier(nn.Module):
    def __init__(self, channel=3, cls_labels=10, size=128):
        super().__init__()
        self.model = nn.Sequential(
                nn.Conv2d(
                    channel,
                    size,
                    kernel_size=3,
                    stride=2,
                    padding=1,
                    bias=False
                ),
                nn.BatchNorm2d(size),
                nn.ReLU(inplace=True),
                nn.Conv2d(
                    size,
                    size,
                    kernel_size=3,
                    stride=2,
                    padding=1,
                    bias=False
                ),
                nn.BatchNorm2d(size),
                nn.ReLU(inplace=True),
                nn.AvgPool2d(2)
            )
        self.linear = nn.Linear(size, cls_labels)
    
    def forward(self, x):
        out = self.model(x)
        return self.linear(out.view(out.size(0), -1))


@BACKBONES.register_module()
class MultiScaleNet(BaseBackbone):
    def __init__(self,
        growth_rate = 32,
        reduction_rate = 16,
        compress_factor = 0.25,
        depths = [6, 12, 24, 16],
        channels = 16,
        n_scales = 3,
        n_blocks = 2,
        block_step = 2,
        stepmode = "even",
        step = 8,
        bnwidth = [4, 2, 2, 1],
        cls_labels = 10,
    ):
        super().__init__()
        self.n_scales = n_scales
        self.init_layer = InitialLayer(3, channels, n_scales=n_scales)
        
        self.classifier = nn.ModuleList()
        self.scale_flow_list = nn.ModuleList()

        steps = [step]

        # Blocks in every scale, [0, 2, 4, 6, 8]
        self.blocks_per_flow = [0] + [block_step*i+n_blocks for i in range(n_scales)]

        gr = copy.copy(growth_rate)
        # For each scale, 1, 2 .. n-1
        for i in range(n_scales):
            scale_flow = nn.ModuleList()

            mul = 2 ** (n_scales - i - 1)
            
            in_c = channels * (mul)
            in_c_lf = []
            block = 1
            
            for j in range(self.blocks_per_flow[i+1]):
                growth_rate = gr * mul

                transition = False

                for k in range(i):
                    chk = torch.floor(torch.tensor(((k+1) * self.blocks_per_flow[i+1]) / i+1))
                    if self.blocks_per_flow[i+1] == chk:
                        transition = True

                if block > self.blocks_per_flow[i]:
                    
                    denseblock = DenseBlock(
                        steps[block-1],
                        in_c,
                        growth_rate,
                        reduction_rate,
                        transition,
                        bnwidth[i]
                    )

                    out_channels = []
                    
                    for k in range(steps[block-1]+1):
                        out_c = (in_c + k * growth_rate)
                        out_channels.append(out_c)
                    
                    if transition:
                        out_c = torch.floor(torch.tensor(1.0 * reduction_rate * out_c))
                    out_channels.append(out_c)

                    # Last step + Current Step
                    if stepmode == "lg":
                        steps.append(steps[-1] + step)
                    # Repeat same step size
                    else:
                        steps.append(step)
                else:
                    
                    prev = self.blocks_per_flow[:i+1]
                    if block in prev[-(i):]:
                        down = 'same'
                    else:
                        down = 'down'

                    denseblock = DenseFusionBlock(
                        steps[block-1],
                        in_c,
                        in_c_lfs[j],
                        growth_rate,
                        reduction_rate,
                        transition,
                        compress_factor,
                        bnwidth[i],
                        bnwidth[i-1],
                        down_type=down
                    )

                    out_channels = []
                    for k in range(steps[block-1]+1):
                        out_c = (in_c + k * growth_rate)
                        out_channels.append(out_c)
                    
                    if transition:
                        out_c = torch.floor(torch.tensor(1.0 * reduction_rate * out_c))
                    out_channels.append(out_c)
                
                scale_flow.append(denseblock)
                in_c = out_channels[-1]
                in_c_lf.append(out_channels[:-1])

                if block > self.blocks_per_flow[i]:
                    self.classifier.append(
                        Classifier(channel=in_c, cls_labels=cls_labels)
                    )
                block += 1

            in_c_lfs = in_c_lf
            self.scale_flow_list.append(scale_flow)

        self.n_exits = len(self.classifier)

        for sf in self.scale_flow_list:
            for m in sf.modules():
                self._init_weights(m)

        for cl in self.classifier:
            for m in cl.modules():
                self._init_weights(m)

    def _init_weights(self, layer):
        if isinstance(layer, nn.Conv2d):
            n = layer.kernel_size[0] * layer.kernel_size[1] * layer.out_channels
            layer.weight.data.normal_(0, torch.sqrt(torch.tensor(2./n)))
        elif isinstance(layer, nn.BatchNorm2d):
            layer.weight.data.fill_(1)
            layer.bias.data.zero_()
        elif isinstance(layer, nn.Linear):
            layer.bias.data.zero_()

    def forward(self, x):
        scales = self.init_layer(x)

        out, lfs = [], []
        cls_id = 0
        for i in range(self.n_scales):
            temp_out = scales[i]
            temp_lfs = []
            block = 0
            for j in range(self.blocks_per_flow[i+1]):
                if self.scale_flow_list[i][j].type == "basic":
                    temp_out, temp_lf = self.scale_flow_list[i][j](temp_out)
                else:
                    temp_out, temp_lf = self.scale_flow_list[i][j](temp_out, lfs[j])
                temp_lfs.append(temp_lf)
                block += 1

                if block > self.blocks_per_flow[i]:
                    out.append(self.classifier[cls_id](temp_out))
                    cls_id += 1
            
            lfs = temp_lfs
        return out