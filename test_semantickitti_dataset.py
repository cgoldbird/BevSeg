# Copyright (c) OpenMMLab. All rights reserved.
import unittest

import numpy as np

from mmdet3d.datasets import SemanticKittiDataset
from mmdet3d.utils import register_all_modules
from mmengine.config import Config

def _generate_semantickitti_dataset_config():
    data_root = './semantickitti/'
    ann_file = 'semantickitti_infos.pkl'
    classes = ('car', 'bicycle', 'motorcycle', 'truck', 'bus', 'person',
               'bicyclist', 'motorcyclist', 'road', 'parking', 'sidewalk',
               'other-ground', 'building', 'fence', 'vegetation', 'trunck',
               'terrian', 'pole', 'traffic-sign')

    seg_label_mapping = {
        0: 19,  # "unlabeled"
        1: 19,  # "outlier" mapped to "unlabeled" --------------mapped
        10: 0,  # "car"
        11: 1,  # "bicycle"
        13: 4,  # "bus" mapped to "other-vehicle" --------------mapped
        15: 2,  # "motorcycle"
        16: 4,  # "on-rails" mapped to "other-vehicle" ---------mapped
        18: 3,  # "truck"
        20: 4,  # "other-vehicle"
        30: 5,  # "person"
        31: 6,  # "bicyclist"
        32: 7,  # "motorcyclist"
        40: 8,  # "road"
        44: 9,  # "parking"
        48: 10,  # "sidewalk"
        49: 11,  # "other-ground"
        50: 12,  # "building"
        51: 13,  # "fence"
        52: 19,  # "other-structure" mapped to "unlabeled" ------mapped
        60: 8,  # "lane-marking" to "road" ---------------------mapped
        70: 14,  # "vegetation"
        71: 15,  # "trunk"
        72: 16,  # "terrain"
        80: 17,  # "pole"
        81: 18,  # "traffic-sign"
        99: 19,  # "other-object" to "unlabeled" ----------------mapped
        252: 0,  # "moving-car" to "car" ------------------------mapped
        253: 6,  # "moving-bicyclist" to "bicyclist" ------------mapped
        254: 5,  # "moving-person" to "person" ------------------mapped
        255: 7,  # "moving-motorcyclist" to "motorcyclist" ------mapped
        256: 4,  # "moving-on-rails" mapped to "other-vehic------mapped
        257: 4,  # "moving-bus" mapped to "other-vehicle" -------mapped
        258: 3,  # "moving-truck" to "truck" --------------------mapped
        259: 4  # "moving-other"-vehicle to "other-vehicle"-----mapped
    }
    max_label = 259
    modality = dict(use_lidar=True, use_camera=False)
    pipeline = [
        dict(
            type='LoadPointsFromFile',
            coord_type='LIDAR',
            shift_height=True,
            load_dim=4,
            use_dim=[0, 1, 2]),
        dict(
            type='LoadAnnotations3D',
            with_bbox_3d=False,
            with_label_3d=False,
            with_mask_3d=False,
            with_seg_3d=True,
            seg_3d_dtype='np.int32'),
        dict(type='PointSegClassMapping'),
        dict(
            type='RandomFlip3D',
            sync_2d=False,
            flip_ratio_bev_horizontal=0.5,
            flip_ratio_bev_vertical=0.5),
        dict(
            type='GlobalRotScaleTrans',
            rot_range=[-3.1415926, 3.1415926],
            scale_ratio_range=[0.95, 1.05],
            translation_std=[0.1, 0.1, 0.1]),
        dict(
            type='SemkittiRangeView',
            H=64,
            W=512,
            fov_up=3.0,
            fov_down=-25.0,
            means=(11.71279, -0.1023471, 0.4952, -1.0545, 0.2877),
            stds=(10.24, 12.295865, 9.4287, 0.8643, 0.1450),
            ignore_index=19),
        dict(type='Pack3DDetInputs', keys=['img', 'gt_semantic_seg'])
    ]

    data_prefix = dict(
        pts='sequences/00/velodyne', pts_semantic_mask='sequences/00/labels')

    return (data_root, ann_file, classes, data_prefix, pipeline, modality,
            seg_label_mapping, max_label)


class TestSemanticKittiDataset(unittest.TestCase):

    def test_semantickitti(self):
        (data_root, ann_file, classes, data_prefix, pipeline, modality,
         seg_label_mapping,
         max_label) = _generate_semantickitti_dataset_config()

        register_all_modules()
        cfg = Config.fromfile('custom_imports.py')
        np.random.seed(0)
        semantickitti_dataset = SemanticKittiDataset(
            data_root,
            ann_file,
            metainfo=dict(
                classes=classes,
                seg_label_mapping=seg_label_mapping,
                max_label=max_label),
            data_prefix=data_prefix,
            pipeline=pipeline,
            modality=modality)

        input_dict = semantickitti_dataset.prepare_data(0)

        range_img = input_dict['inputs']['img']
        data_sample = input_dict['data_samples']
        range_label = data_sample.gt_pts_seg.semantic_seg
        print(range_img.shape)
        print(range_label.shape)
        
if __name__ == "__main__":
    unittest.main()
