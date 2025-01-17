from __future__ import absolute_import
from __future__ import unicode_literals
from __future__ import print_function
from __future__ import division

import torch
import torch.nn as nn
from torch.autograd import Variable
from functools import reduce
import operator

class FlopCounter():
    def __init__(self):
        self.reset()

    def reset(self):
        self.count_ops = 0
        self.count_params = 0
        self.cls_ops = []
        self.cls_params = []
        self.special_flops = []

    def get_total_flops(self):
        return self.count_ops

    def get_special_flops(self):
        return self.special_flops

    def get_num_gen(self, gen):
        return sum(1 for x in gen)

    def is_leaf(self, model):
        return self.get_num_gen(model.children()) == 0

    def get_layer_info(self, layer):
        layer_str = str(layer)
        type_name = layer_str[:layer_str.find('(')].strip()
        return type_name

    def get_layer_param(self, model):
        return sum([reduce(operator.mul, i.size(), 1) for i in model.parameters()])

    ### The input batch size should be 1 to call this function
    def measure_layer(self, layer, x):
        delta_ops = 0
        delta_params = 0
        multi_add = 1
        type_name = self.get_layer_info(layer)

        if type_name in ['CGConv2d']:
            out_h = int((x.size()[2] + 2 * layer.padding[0] - layer.kernel_size[0]) /
                        layer.stride[0] + 1)
            out_w = int((x.size()[3] + 2 * layer.padding[1] - layer.kernel_size[1]) /
                        layer.stride[1] + 1)
            delta_ops = layer.in_channels * layer.out_channels * layer.kernel_size[0] *  \
                    layer.kernel_size[1] * out_h * out_w / layer.groups * multi_add
            delta_params = self.get_layer_param(layer)
            self.special_flops.append(delta_ops)

        elif type_name in ['Conv2d']:
            out_h = int((x.size()[2] + 2 * layer.padding[0] - layer.kernel_size[0]) /
                        layer.stride[0] + 1)
            out_w = int((x.size()[3] + 2 * layer.padding[1] - layer.kernel_size[1]) /
                        layer.stride[1] + 1)
            delta_ops = layer.in_channels * layer.out_channels * layer.kernel_size[0] *  \
                    layer.kernel_size[1] * out_h * out_w / layer.groups * multi_add
            delta_params = self.get_layer_param(layer)

        elif type_name in ['ReLU']:
            delta_ops = x.numel()
            delta_params = self.get_layer_param(layer)

        elif type_name in ['AvgPool2d', 'MaxPool2d']:
            in_w = x.size()[2]
            kernel_ops = layer.kernel_size * layer.kernel_size
            out_w = int((in_w + 2 * layer.padding - layer.kernel_size) / layer.stride + 1)
            out_h = int((in_w + 2 * layer.padding - layer.kernel_size) / layer.stride + 1)
            delta_ops = x.size()[0] * x.size()[1] * out_w * out_h * kernel_ops
            delta_params = self.get_layer_param(layer)

        elif type_name in ['AdaptiveAvgPool2d']:
            delta_ops = x.size()[0] * x.size()[1] * x.size()[2] * x.size()[3]
            delta_params = self.get_layer_param(layer)

        elif type_name in ['Linear']:
            weight_ops = layer.weight.numel() * multi_add
            bias_ops = layer.bias.numel()
            delta_ops = x.size()[0] * (weight_ops + bias_ops)
            delta_params = self.get_layer_param(layer)

        elif type_name in ['BatchNorm2d', 'Dropout2d', 'DropChannel', 'Dropout',
                        'MSDNFirstLayer', 'ConvBasic', 'ConvBN',
                        'ParallelModule', 'MSDNet', 'Sequential',
                        'MSDNLayer', 'ConvDownNormal', 'ConvNormal', 'ClassifierModule',
                        'Grayscale', 'Identity', 'Flatten', 'Softmax']:
            delta_params = self.get_layer_param(layer)

        ### unknown layer type
        else:
            raise TypeError('unknown layer type: %s' % type_name)

        self.count_ops += delta_ops
        self.count_params += delta_params
        if type_name == 'Linear':
            print('---------------------')
            print(layer)
            print('FLOPs: %.2fM, Params: %.2fM' % (self.count_ops / 1e6, self.count_params / 1e6))
            self.cls_ops.append(self.count_ops)
            self.cls_params.append(self.count_params)
        return

    def is_special(self, child):
        return self.get_layer_info(child) in {'CGConv2d'}

    def measure_model(self, model, H, W):
        self.reset()
        data = Variable(torch.zeros(1, 3, H, W))

        def should_measure(x):
            return self.is_leaf(x) or self.is_special(x)

        def modify_forward(model):
            for child in model.children():
                if should_measure(child):
                    def new_forward(m):
                        def lambda_forward(x):
                            self.measure_layer(m, x)
                            return m.old_forward(x)
                        return lambda_forward
                    child.old_forward = child.forward
                    child.forward = new_forward(child)
                else:
                    modify_forward(child)

        def restore_forward(model):
            for child in model.children():
                # leaf node
                if self.is_leaf(child) and hasattr(child, 'old_forward'):
                    child.forward = child.old_forward
                    child.old_forward = None
                else:
                    restore_forward(child)

        model.eval()
        modify_forward(model)
        model.forward(data)
        restore_forward(model)
        if self.cls_ops == []:
            self.cls_ops.append(self.count_ops)
            self.cls_params.append(self.count_params)
        return self.cls_ops, self.cls_params
