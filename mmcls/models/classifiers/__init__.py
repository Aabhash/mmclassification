# Copyright (c) OpenMMLab. All rights reserved.
from .base import BaseClassifier
from .image import ImageClassifier
from .example import ExampleClassifier #import class of classifier 
from .multi_scale import MultiScaleClassifier
from .cascading import Cascading
from .cgnet import CGClassifier 
from .earlyExit import EarlyExitClassifier

__all__ = ['BaseClassifier', 'ImageClassifier', "ExampleClassifier", 'Cascading', 'CGClassifier', "MultiScaleClassifier",  'EarlyExitClassifier']    #add classifier
