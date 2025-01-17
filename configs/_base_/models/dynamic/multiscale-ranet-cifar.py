# model settings
model = dict(
    type='MultiScaleClassifier',
    backbone=dict(
        type='MultiScaleNetCifar',
        growth_rate=8,
        reduction_rate=0.5,
        compress_factor=0.25,
        channels=32,
        n_scales=3,
        n_blocks=2,
        block_step=2,
        stepmode="even",
        step=4,
        bnwidth=[4, 2, 1],
        cls_labels=10,
    ),
    get_infos='./results/multiscale-ranet-cifar/',
    neck=None,
    head=dict(
        type='MultiScaleHead',
        # num_classes=10,
        # in_channels=1280,
        loss=dict(type='CrossEntropyLoss', loss_weight=1.0),
        topk=(1, ),
        num_exits=6
    ),
    pretrained=None
)
