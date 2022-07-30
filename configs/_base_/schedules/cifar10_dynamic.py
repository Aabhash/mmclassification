# optimizer
optimizer = dict(type='SGD', lr=0.1, momentum=0.9, weight_decay=0.0001)
optimizer_config = dict(grad_clip=None)
# learning policy
# lr_config = dict(policy='step', step=[25, 50, 100, 125, 150])
lr_config = dict(
    policy='CosineAnnealing',
    min_lr=0,
    warmup='linear',
    warmup_iters=1000,
    warmup_ratio=0.25)
runner = dict(type='EpochBasedRunner', max_epochs=200)