from tokenize import Special
from numpy import block


model = dict(
    type='ImageClassifier_dynamic',
    backbone=dict(
        type='ResNetFeedForwardSP',
        layers=[2,2,2,2],
        gate_type= 'ffgate1',
        ),
    special = "skip",
    head=dict(
        type='LinearClsHead',
        num_classes=10,
        in_channels=4096,
        loss=dict(type='CrossEntropyLoss', loss_weight=1.0),

    ),

  
)
"""
    
"""