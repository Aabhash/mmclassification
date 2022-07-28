from numpy import block


model = dict(
    type='ImageClassifier',
    backbone=dict(
        type='ResNetFeedForwardSP_cifar',
        layers=[6,6,6],
        gate_type= 'ffgate2'
        ),

    head=dict(
        type='LinearClsHead',
        num_classes=10,
        in_channels=64,
        loss=dict(type='CrossEntropyLoss', loss_weight=1.0),

    ),


)