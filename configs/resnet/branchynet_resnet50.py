_base_ = 'resnet50_8xb16_cifar10.py'

model = dict(
    # type='ImageClassifier',
    backbone=dict(
        type='res_inter_classifiers'),
)
    