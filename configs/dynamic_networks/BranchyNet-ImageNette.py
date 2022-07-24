model = dict(
    type='EarlyExitClassifier',
    backbone=dict(
        type='BranchyNetImagenette',
        activated_branches=[True, True, True]),
    head=dict(
        type='emptyClsHead',     # linear classification head，
        loss=dict(type='BranchyNetLoss'), # Loss function configuration information
        topk=(1, 5),              # Evaluation index, Top-k accuracy rate, here is the accuracy rate of top1 and top5
    )
    
)
# dataset settings
dataset_type = 'ImageNette'
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='RandomResizedCrop', size=224),
    dict(type='RandomFlip', flip_prob=0.5, direction='horizontal'),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='ImageToTensor', keys=['img']),
    dict(type='ToTensor', keys=['gt_label']),
    dict(type='Collect', keys=['img', 'gt_label'])
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='Resize', size=(256, -1)),
    dict(type='CenterCrop', crop_size=224),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='ImageToTensor', keys=['img']),
    dict(type='Collect', keys=['img'])
]
data = dict(
    samples_per_gpu=32,
    workers_per_gpu=1,
    train=dict(
        type=dataset_type,
        data_prefix='data/imagenette2/train',
        pipeline=train_pipeline),
    val=dict(
        type=dataset_type,
        data_prefix='data/imagenette2/val',
        pipeline=test_pipeline),
    test=dict(
        type=dataset_type,
        data_prefix='data/cifar10',
        pipeline=test_pipeline,
        test_mode=True),
)
# he configuration file used to build the optimizer, support all optimizers in PyTorch.
optimizer = dict(type='SGD',         # Optimizer type
                lr=0.001,              # Learning rate of optimizers, see detail usages of the parameters in the documentation of PyTorch
                momentum=0.9,        # Momentum
                weight_decay=0.0001) # Weight decay of SGD
# Config used to build the optimizer hook, refer to https://github.com/open-mmlab/mmcv/blob/master/mmcv/runner/hooks/optimizer.py#L8 for implementation details.
optimizer_config = dict(grad_clip=None)  # Most of the methods do not use gradient clip
# Learning rate scheduler config used to register LrUpdater hook
lr_config = dict(policy='step',          # The policy of scheduler, also support CosineAnnealing, Cyclic, etc. Refer to details of supported LrUpdater from https://github.com/open-mmlab/mmcv/blob/master/mmcv/runner/hooks/lr_updater.py#L9.
                 step=[10, 20, 30])      # Steps to decay the learning rate
runner = dict(type='EpochBasedRunner',   # Type of runner to use (i.e. IterBasedRunner or EpochBasedRunner)
            max_epochs=15)    # Runner that runs the workflow in total max_epochs. For IterBasedRunner use `max_iters`

# Config to set the checkpoint hook, Refer to https://github.com/open-mmlab/mmcv/blob/master/mmcv/runner/hooks/checkpoint.py for implementation.
checkpoint_config = dict(interval=1)    # The save interval is 1
# config to register logger hook
log_config = dict(
    interval=100,                       # Interval to print the log
    hooks=[
        dict(type='TextLoggerHook'),           # The Tensorboard logger is also supported
        # dict(type='TensorboardLoggerHook')
    ])

dist_params = dict(backend='nccl')   # Parameters to setup distributed training, the port can also be set.
log_level = 'INFO'             # The output level of the log.
resume_from = None             # Resume checkpoints from a given path, the training will be resumed from the epoch when the checkpoint's is saved.
workflow = [('train', 1)]      # Workflow for runner. [('train', 1)] means there is only one workflow and the workflow named 'train' is executed once.
work_dir = 'work_dir'          # Directory to save the model checkpoints and logs for the current experiments

load_from = False