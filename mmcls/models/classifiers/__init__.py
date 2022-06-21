# Copyright (c) OpenMMLab. All rights reserved.
from .base import BaseClassifier
from .image import ImageClassifier
from .example import ExampleClassifier #import class of classifier 

__all__ = ['BaseClassifier', 'ImageClassifier', "ExampleClassifier"]    #add classifier
