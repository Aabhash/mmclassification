# model settings
model = dict(
    type='MultiScaleClassifier',
    backbone=dict(
        type='MultiScaleNet',
        growth_rate=16,
        reduction_rate=0.5,
        compress_factor=0.25,
        channels=32,
        n_scales=4,
        n_blocks=2,
        block_step=2,
        stepmode="even",
        step=8,
        bnwidth=[4, 2, 2, 1],
        gr_factor=[1, 2, 2, 4],
        cls_labels=10,
    ),
    get_infos='./results/multiscale-ranet/exits.json',
    neck=None,
    head=dict(
        type='MultiScaleHead',
        # num_classes=10,
        # in_channels=1280,
        loss=dict(type='CrossEntropyLoss', loss_weight=1.0),
        topk=(1, ),
        num_exits=8
    ),
    pretrained=None
)
