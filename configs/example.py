_base_ = [
    '../configs/_base_/models/example.py', '../configs/_base_/datasets/cifar10_bs16.py',
    '../configs/_base_/schedules/cifar10_bs128.py', '../configs/_base_/default_runtime.py'
]
# first is path to model config
# second is path to dataset
# thirt, fourth I don't  right now