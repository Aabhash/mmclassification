from ..builder import CLASSIFIERS, build_backbone, build_head, build_neck
from ..heads import MultiLabelClsHead
from ..utils.augment import Augments
from .base import BaseClassifier


@CLASSIFIERS.register_module()
class ExampleClassifier(BaseClassifier):
    # these are also the key words for the config
    
    def __init__(self,
                 exampleNet1,
                 exampleNet2=None,
                 exampleHead1 = None,
                 examppleHead2 = None,
                 exampleVariable = None,
                 init_cfg=None):
        super(ExampleClassifier, self).__init__(init_cfg)
        self.exampleVariable = exampleVariable
        self.exampleNet1 = build_backbone(exampleNet1)
        if exampleHead1:
            self.exampleHead1 = build_head(exampleHead1)
    
    def extract_feat(self, imgs, stage=None):
        x = self.exampleNet1(imgs)

        if stage == 'backbone':
            return x

        
        if stage == 'neck':
            return self.exampleHead1(x)

    def forward_train(self, imgs, **kwargs):
        """
            train logik for network
        """
        pass

    def simple_test(self, img, **kwargs):
        """
            test logik for network
        """
        pass