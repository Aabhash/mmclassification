_base_ = [
    '../_base_/models/dynamic/cgnet-cifar.py',
    '../_base_/datasets/cifar10_bs16.py',
    '../_base_/schedules/cifar10_dynamic.py',
    '../_base_/default_runtime.py'
]