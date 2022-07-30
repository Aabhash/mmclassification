# model settings
# model = dict(
#     type='CGClassifier',
#     backbone=dict(
#         type='CGResNet',
#         num_blocks = [3, 3, 3],
#         num_classes = 10,
#         partitions = 4,
#         ginit = 0.0,
#         gtarget=1.0,
#         use_group = False,
#         sparse_bp = False,
#         shuffle = False,
#         alpha = 2.0
#     ),
#     neck=None,
#     head=dict(
#         type='LinearClsHead',
#         num_classes=10,
#         in_channels=64,
#         loss=dict(type='CrossEntropyLoss', loss_weight=1.0),

#     ),
#     pretrained=None
# )
# model settings
model = dict(
    type='CGClassifier',
    backbone=dict(
        type='CGResNet',
        depth = 18,
        num_blocks = [2, 2, 2],
        in_channels = 3,
        base_channels = 16,
        partitions = 2,
        ginit = 0.0,
        gtarget=2.0,
        alpha = 4.0
    ),
    neck=None,
    head=dict(
        type='LinearClsHead',
        num_classes=10,
        in_channels=64,
        loss=dict(type='CrossEntropyLoss', loss_weight=1.0),

    ),
    pretrained=None
)
