_base_ = [
    '../_base_/models/dynamic/multiscale-ranet-cifar.py',
    '../_base_/datasets/cifar10_bs256.py',
    # '../_base_/schedules/cifar10_bs128_dn.py',
    '../_base_/schedules/cifar10_dynamic.py',
    '../_base_/default_runtime.py'
]