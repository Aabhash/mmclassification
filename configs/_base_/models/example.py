# model settings
model = dict(
    type='ExampleClassifier',
    exampleNet1=dict(
        type='ResNet_CIFAR',
        depth=18,
        num_stages=4,
        out_indices=(3, ),
        style='pytorch'),
    
    exampleHead1=dict(
        type='LinearClsHead',
        num_classes=10,
        in_channels=512,
        loss=dict(type='CrossEntropyLoss', loss_weight=1.0),
    ))
