_base_ = [
    '../_base_/models/resnet50.py', '../_base_/datasets/imagenet_small.py',
    '../_base_/schedules/imagenet_bs256_200e_coslr_warmup.py', '../_base_/default_runtime.py'
] 