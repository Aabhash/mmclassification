model = dict(
    type='Cascading',
    little=dict(
        type='ResNet_CIFAR',
        depth=18,
        num_stages=4,
        out_indices=(3, ),
        style='pytorch'),
    little_neck=dict(type='GlobalAveragePooling'),
    little_head=dict(
        type='LinearClsHead',
        num_classes=10,
        in_channels=512,
        loss=dict(type='CrossEntropyLoss', loss_weight=1.0),
    ),
    big=dict(
        type='ResNet_CIFAR',
        depth=50,
        num_stages=4,
        out_indices=(3, ),
        style='pytorch'),
    big_neck=dict(type='GlobalAveragePooling'),
    big_head=dict(
        type='LinearClsHead',
        num_classes=10,
        in_channels=2048,
        loss=dict(type='CrossEntropyLoss', loss_weight=1.0),

    ),

    pretrained_little = '/home/till/mmclassification/pretrained_models/resnet18_b16x8_cifar10_20210528-bd6371c8.pth',
    pretrained_big = "/home/till/mmclassification/pretrained_models/resnet50_b16x8_cifar10_20210528-f54bfad9.pth"
)
"""
    
"""