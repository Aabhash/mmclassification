model = dict(
    type='MOE',
    n_experts = 5,
    expert_backbone=dict(
        type='DenseNet',
        arch='expert'),
    expert_neck=dict(type='GlobalAveragePooling'),
   
    gate_network=dict(
        type='DenseNet',
        arch='gate'),
    gate_neck=dict(type='GlobalAveragePooling'),
    end_head=dict(
        type='MixtureClsHead',
        num_classes=10,
        in_channels=256,
        n_experts= 5,
        loss=dict(type='ExpertProbabiltityLoss', num_classes= 10),
    )

    )

   
