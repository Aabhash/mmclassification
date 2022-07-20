# Copyright (c) OpenMMLab. All rights reserved.
from .base import BaseClassifier
from .image import ImageClassifier
from .example import ExampleClassifier #import class of classifier 
from .cascading import Cascading
from. earlyExit import EarlyExitClassifier

__all__ = ['BaseClassifier', 'ImageClassifier', "ExampleClassifier",'Cascading', 'EarlyExitClassifier']    #add classifier


