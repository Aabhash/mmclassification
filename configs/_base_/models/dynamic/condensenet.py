model = dict(
    type='ImageClassifier',
    backbone=dict(type='CondenseNet',
    stages = "14-14-14",
    growth = "8-16-32",
    args = ""),
    head=dict(
        type='LinearClsHead',
        num_classes=10,
        in_channels=800,
        loss=dict(type='CrossEntropyLoss', loss_weight=1.0),
    ))