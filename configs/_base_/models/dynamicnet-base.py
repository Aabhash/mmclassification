# model settings
model = dict(
    type='ImageClassifier',
    backbone=dict(
        type='MultiScaleNet',        
        growth_rate = 6,
        reduction_rate = 0.5,
        compress_factor = 0.25,
        depths = [6, 12, 24, 16],
        channels = 16,
        n_scales = 3,
        n_blocks = 2,
        block_step = 2,
        stepmode = "even",
        step = 4,
        bnwidth = [4, 2, 1],
        cls_labels = 10,
    )
)
