model = dict(
    type='MOE',
    n_experts = 5,
    expert_backbone=dict(
        type='ResNet_CIFAR',
        depth=18,
        num_stages=4,
        out_indices=(3, ),
        style='pytorch'),
    expert_neck=dict(type='GlobalAveragePooling'),
   
    gate_network=dict(
        type='DenseNet',
        arch='gate'),
    gate_neck=dict(type='GlobalAveragePooling'),
    end_head=dict(
        type='MixtureClsHead',
        num_classes=10,
        in_channels=512,
        n_experts= 5,
        in_gate_head = 192, 
        loss=dict(type='ExpertProbabiltityLoss', num_classes= 10),
    )

    )
