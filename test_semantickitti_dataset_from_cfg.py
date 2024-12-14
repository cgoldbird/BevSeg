# Copyright (c) OpenMMLab. All rights reserved.
import unittest

import numpy as np

from mmdet3d.datasets import SemanticKittiDataset
from mmdet3d.utils import register_all_modules
from mmengine.config import Config
from mmengine.registry import DATASETS


class TestSemanticKittiDataset(unittest.TestCase):

    def test_semantickitti(self):
        cfg_imports = Config.fromfile('custom_imports.py')
        register_all_modules()
        np.random.seed(0)
        cfg = Config.fromfile('configs/semantickitti_pipeline_RangeView.py')
        semantickitti_dataset = DATASETS.build(cfg)

        input_dict = semantickitti_dataset.prepare_data(0)

        range_img = input_dict['inputs']['img']
        data_sample = input_dict['data_samples']
        range_label = data_sample.gt_pts_seg.semantic_seg
        print(range_img.shape)
        print(range_label.shape)
        
if __name__ == "__main__":
    unittest.main()
