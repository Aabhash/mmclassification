from ..builder import CLASSIFIERS, build_backbone, build_head, build_neck
from ..heads import MultiLabelClsHead
from ..utils.augment import Augments
from .base import BaseClassifier


@CLASSIFIERS.register_module()
class Test(BaseClassifier):
    # these are also the key words for the config
    
    def __init__(self,
                 test,
                 init_cfg=None):
        super(Test, self).__init__(init_cfg)

        print(test)
    
    def extract_feat(self, imgs, stage=None):
        print("extract_feat")

    def forward_train(self, imgs, **kwargs):
        """
        Args:
            img (list[Tensor]): List of tensors of shape (1, C, H, W).
                Typically these should be mean centered and std scaled.
            kwargs (keyword arguments): Specific to concrete implementation.
        """
        print("forward train")

    def simple_test(self, img, **kwargs):
        pass