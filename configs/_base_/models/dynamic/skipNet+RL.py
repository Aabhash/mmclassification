from numpy import block


model = dict(
    type='ImageClassifier',
    backbone=dict(
        type='ResNetFeedForwardRL',
        layers=[2,2,2,2],
        gate_type= 'ffgate1'
        ),
   
    head=dict(
        type='LinearClsHead',
        num_classes=10,
        in_channels=4096,
        loss=dict(type='CrossEntropyLoss', loss_weight=1.0),

    ),

  
)