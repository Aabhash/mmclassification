# Copyright (c) OpenMMLab. All rights reserved.
from .base import BaseClassifier
from .image import ImageClassifier
from .image_dynmaic import ImageClassifier_dynamic
from .example import ExampleClassifier #import class of classifier 
from .cascading import Cascading

__all__ = ['BaseClassifier', 'ImageClassifier', "ExampleClassifier",'Cascading', 'ImageClassifier_dynamic']    #add classifier


