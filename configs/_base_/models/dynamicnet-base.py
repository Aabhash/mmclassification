# model settings
model = dict(
    type='ImageClassifier',
    backbone=dict(
        type='MultiScaleNet',        
        growth_rate = 32,
        reduction_rate = 16,
        depths = [6, 12, 24, 16],
        channels = 64,
        n_scales = 3,
        n_blocks = 2,
        block_step = 3,
        stepmode = "even",
        step = 8,
        bnwidth = [4, 2, 2, 1],
        cls_labels = 10,
    )
)
