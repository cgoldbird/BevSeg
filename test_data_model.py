# Copyright (c) OpenMMLab. All rights reserved.
import unittest
import numpy as np

import pytest
import torch

from mmdet3d.models.data_preprocessors import Det3DDataPreprocessor
from mmdet3d.structures import Det3DDataSample, PointData
from mmdet3d.utils import register_all_modules
from mmengine.config import Config
from mmengine.registry import DATASETS

from bevseg.data_preprocessors.data_preprocessor import VoxelPreprocessor
from bevseg.voxel_encoders.voxel_encoder import VoxelFeatureEncoder
from bevseg.voxel_encoders.bev_encoder import BevFeatureEncoder
from bevseg.backbone.bev_backbone_2 import Unet

class TestDet3DDataPreprocessor(unittest.TestCase):

    def test_forward(self):
        register_all_modules()
        data_cfg = Config.fromfile('configs/semantickitti_pipeline_Point.py')
        dataset = DATASETS.build(data_cfg)
        data = dataset.prepare_data(0)
        inputs_dict = {'points': [data['inputs']['points']]}
        data = {'inputs': inputs_dict, 'data_samples': [data['data_samples']]}
        
        # Preprocess data
        processor = VoxelPreprocessor(grid_shape = [480, 360, 32],
                                      point_cloud_range=[-50, -50, -3, 50, 50, 1.5],
                                      ignore_index=19,
                                      voxel_type='cubic').cuda()
        # processor = VoxelPreprocessor(grid_shape = [480, 360, 32],
        #                               point_cloud_range=[3, -3.14159265359, -3, 50, 3.14159265359, 1.5],
        #                               ignore_index=19,
        #                               voxel_type='cylindrical').cuda()
        out_data = processor(data)
        seglabel = out_data['data_samples'][0].gt_pts_seg.semantic_seg
        print(seglabel.shape)
        # 保存为点云文件
        # seglabel = seglabel.cpu().numpy()
        # seglabel = seglabel.reshape(-1, 1)
        # seglabel = seglabel.astype(np.int32)
        # seglabel = seglabel.tobytes()
        # with open('seglabel.bin', 'wb') as f:
        #     f.write(seglabel)
        
        # Get voxel features
        voxel_dict = out_data['inputs']['voxels'].copy()
        # voxel_encoder = VoxelFeatureEncoder(in_channels=7,
        #                                     feat_channels=[64, 128, 256],
        #                                     # norm_cfg=dict(type='SyncBN', eps=1e-3, momentum=0.01),
        #                                     with_pre_norm=True,
        #                                     feat_compression=32).cuda()
        voxel_encoder = BevFeatureEncoder(in_channels=7,
                                          feat_channels=[64, 128, 256],
                                          # norm_cfg=dict(type='SyncBN', eps=1e-3, momentum=0.01),
                                          with_pre_norm=True,
                                          feat_compression=32).cuda()
        voxel_dict = voxel_encoder(voxel_dict)
        # backbone
        backbone = Unet().cuda()
        voxel_dict = backbone(voxel_dict) 
        print(voxel_dict['bev_feats_map'].shape)
        
        
if __name__ == "__main__":
    unittest.main()