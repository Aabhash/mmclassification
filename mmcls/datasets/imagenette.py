# Copyright (c) OpenMMLab. All rights reserved.
from typing import Optional, Sequence, Union

from .builder import DATASETS
from .custom import CustomDataset


@DATASETS.register_module()
class ImageNette(CustomDataset):
    """ Copied from Tills Branch"""
      # noqa: E501

    IMG_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tif')
    CLASSES = [
       'tench', 'English springer', 'cassette player', 'chain saw', 'church', 'French horn', 'garbage truck', 'gas pump', 'golf ball', 'parachute'
    ]

    def __init__(self,
                 data_prefix: str,
                 pipeline: Sequence = (),
                 classes: Union[str, Sequence[str], None] = None,
                 ann_file: Optional[str] = None,
                 test_mode: bool = False,
                 file_client_args: Optional[dict] = None):
        super().__init__(
            data_prefix=data_prefix,
            pipeline=pipeline,
            classes=classes,
            ann_file=ann_file,
            extensions=self.IMG_EXTENSIONS,
            test_mode=test_mode,
            file_client_args=file_client_args)
