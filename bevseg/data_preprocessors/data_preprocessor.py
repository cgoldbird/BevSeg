from typing import List

import numpy as np
import torch
import torch.nn.functional as F
from mmdet3d.registry import MODELS
from mmdet3d.structures.det3d_data_sample import SampleList
from mmengine.model import BaseDataPreprocessor
from torch import Tensor
from .voxelize import VoxelizationByGridShape, dynamic_scatter_3d

@MODELS.register_module()
class VoxelPreprocessor(BaseDataPreprocessor):
    def __init__(self, 
                 grid_shape: List[int],
                 point_cloud_range: List[float],
                 ignore_index: int,
                 non_blocking: bool = False) -> None:
        super(VoxelPreprocessor, self).__init__(non_blocking=non_blocking)
        self.grid_shape = grid_shape
        self.point_cloud_range = point_cloud_range
        self.ignore_index = ignore_index
    
    def forward(self, data: dict, training: bool = False) -> dict:
        data = self.cast_data(data)
        data.setdefault('data_samples', None)
        inputs, data_samples = data['inputs'], data['data_samples']
        batch_inputs = dict()
        
        assert 'points' in inputs, 'points must be in inputs'
        batch_inputs['points'] = inputs['points']
        voxel_dict = self.voxel_group(batch_inputs['points'], data_samples)
        batch_inputs['voxels'] = voxel_dict
        return {'inputs': batch_inputs, 'data_samples': data_samples}
    
    @torch.no_grad()
    def voxel_group(self, points: List[Tensor],
                             data_samples: SampleList) -> dict:
        voxel_dict = dict()
        
        coors = []
        voxels = []
        
        for i, res in enumerate(points):
            xyz_res = res[:, :3]
            max_bound = res.new_tensor(self.point_cloud_range[3:])
            min_bound = res.new_tensor(self.point_cloud_range[:3])
            crop_range = max_bound - min_bound
            cur_grid_shape = res.new_tensor(self.grid_shape)
            intervals = crop_range / (cur_grid_shape - 1)
            
            xyz_res_clamp = torch.clamp(xyz_res, min_bound, max_bound) # PyTorch >= 1.9.0
            
            res_coors = torch.floor((xyz_res_clamp - min_bound) / intervals).int()
            if 'pts_semantic_mask' in data_samples[i].gt_pts_seg:
                pts_semantic_mask = data_samples[i].gt_pts_seg.pts_semantic_mask
                voxel_semantic_mask, voxels_pos, point2voxel_map = dynamic_scatter_3d(
                F.one_hot(pts_semantic_mask.long()).float(), res_coors, 'mean',
                True)
                # torch 1.10.0 has a bug in scatter_nd, so we use the following code
                voxels_pos = voxels_pos.to(torch.int64)
                voxel_semantic_mask = torch.argmax(voxel_semantic_mask, dim=-1)
                seg_label = torch.ones(
                    self.grid_shape,
                    dtype=torch.long,
                    device=pts_semantic_mask.device) * self.ignore_index
                seg_label[voxels_pos[:, 0], voxels_pos[:, 1], voxels_pos[:, 2]] = voxel_semantic_mask
                data_samples[i].gt_pts_seg.semantic_seg = seg_label
            
            voxel_centers = (res_coors.float() + 0.5) * intervals + min_bound
            voxel_xyz_res = xyz_res - voxel_centers # relative coordinates of points to voxel centers
            res_coors = F.pad(res_coors, (1, 0), mode='constant', value=i) # voxel index of points (including batch index at first dim)
            res_feas = torch.cat([xyz_res, voxel_xyz_res, res[:, 3:]], dim=-1) # features of points
            coors.append(res_coors)
            voxels.append(res_feas)
            
        voxels = torch.cat(voxels, dim=0)
        coors = torch.cat(coors, dim=0)
        voxel_dict['voxels'] = voxels
        voxel_dict['coors'] = coors

        return voxel_dict