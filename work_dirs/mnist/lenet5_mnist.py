model = dict(
    type='ImageClassifier',
    backbone=dict(type='LeNet5', num_classes=10),
    neck=None,
    head=dict(
        type='ClsHead', loss=dict(type='CrossEntropyLoss', loss_weight=1.0)))
dataset_type = 'MNIST'
img_norm_cfg = dict(mean=[33.46], std=[78.87], to_rgb=True)
train_pipeline = [
    dict(type='Resize', size=32),
    dict(type='Normalize', mean=[33.46], std=[78.87], to_rgb=True),
    dict(type='ImageToTensor', keys=['img']),
    dict(type='ToTensor', keys=['gt_label']),
    dict(type='Collect', keys=['img', 'gt_label'])
]
test_pipeline = [
    dict(type='Resize', size=32),
    dict(type='Normalize', mean=[33.46], std=[78.87], to_rgb=True),
    dict(type='ImageToTensor', keys=['img']),
    dict(type='Collect', keys=['img'])
]
data = dict(
    samples_per_gpu=128,
    workers_per_gpu=2,
    train=dict(
        type='MNIST',
        data_prefix='data/mnist',
        pipeline=[
            dict(type='Resize', size=32),
            dict(type='Normalize', mean=[33.46], std=[78.87], to_rgb=True),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='ToTensor', keys=['gt_label']),
            dict(type='Collect', keys=['img', 'gt_label'])
        ]),
    val=dict(
        type='MNIST',
        data_prefix='data/mnist',
        pipeline=[
            dict(type='Resize', size=32),
            dict(type='Normalize', mean=[33.46], std=[78.87], to_rgb=True),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img'])
        ]),
    test=dict(
        type='MNIST',
        data_prefix='data/mnist',
        pipeline=[
            dict(type='Resize', size=32),
            dict(type='Normalize', mean=[33.46], std=[78.87], to_rgb=True),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img'])
        ]))
evaluation = dict(
    interval=5, metric='accuracy', metric_options=dict(topk=(1, )))
optimizer = dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0001)
optimizer_config = dict(grad_clip=None)
lr_config = dict(policy='step', step=[15])
checkpoint_config = dict(interval=1)
log_config = dict(interval=150, hooks=[dict(type='TextLoggerHook')])
runner = dict(type='EpochBasedRunner', max_epochs=5)
dist_params = dict(backend='nccl')
log_level = 'INFO'
work_dir = './work_dirs/mnist/'
load_from = None
resume_from = None
workflow = [('train', 1)]
gpu_ids = [0]
