_base_= '.resnet50_8xb32_in1k.py'

runner = dict(max_epochs=300)
lr_config = dict(step = 150, 200, 250)
data = dict(
    train=dict(data_prefix='mydata/imagenet/train'),
    val=dict(data_prefix='mydata/imagenet/train'),
    test=dict(data_prefix='mydata/imagenet/train')
)