type='SemanticKittiDataset'
ann_file='semantickitti_infos.pkl'
data_root='semantickitti/'

metainfo = dict(
    classes = [
    'car', 'bicycle', 'motorcycle', 'truck', 'bus', 'person', 'bicyclist',
    'motorcyclist', 'road', 'parking', 'sidewalk', 'other-ground', 'building',
    'fence', 'vegetation', 'trunck', 'terrian', 'pole', 'traffic-sign'
    ], 
    seg_label_mapping={
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
        }, 
    max_label=259
    )

data_prefix = dict(
        pts='sequences/00/velodyne', pts_semantic_mask='sequences/00/labels')


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
        dict(type='Pack3DDetInputs', keys=['points', 'pts_semantic_mask'])
    ]



modality= dict(use_lidar=True, use_camera=False)