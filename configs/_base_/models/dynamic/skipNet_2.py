model = dict(
    type='ImageClassifier',
    backbone=dict(
        type='RecurrentGatedResNet',
        layers=[3,4,6,3],
        gate_type= 'rnn',
        ),
    head=dict(
        type='LinearClsHead',
        num_classes=10,
        in_channels=512,
        loss=dict(type='CrossEntropyLoss', loss_weight=1.0),

    ),

  
)