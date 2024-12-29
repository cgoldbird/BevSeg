from typing import Optional, Sequence

import torch
import torch.nn as nn
import torch_scatter
from mmcv.cnn import build_norm_layer
from mmdet3d.registry import MODELS
from mmdet3d.utils import ConfigType


@MODELS.register_module()
class BevFeatureEncoder(nn.Module):
    def __init__(self,
                 in_channels: int = 4,
                 feat_channels: Sequence[int] = [64, 128, 256],
                 grid_shape: Sequence[int] = [480, 360, 32],
                 norm_cfg: ConfigType = dict(
                     type='BN1d', eps=1e-5, momentum=0.1),
                 with_pre_norm: bool = False,
                 feat_compression: Optional[int] = None):
        super(BevFeatureEncoder, self).__init__()
        assert len(feat_channels) > 0
        self.in_channels = in_channels
        self.grid_shape = grid_shape
        
        feat_channels = [self.in_channels] + list(feat_channels)
        if with_pre_norm:
            self.pre_norm = build_norm_layer(norm_cfg, self.in_channels)[1]
        else:
            self.pre_norm = None

        ffe_layers = []
        for i in range(len(feat_channels) - 1):
            in_filters = feat_channels[i]
            out_filters = feat_channels[i + 1]
            norm_layer = build_norm_layer(norm_cfg, out_filters)[1]
            ffe_layers.append(
                nn.Sequential(
                    nn.Linear(in_filters, out_filters, bias=False),
                    norm_layer, nn.ReLU(inplace=True)))
        self.ffe_layers = nn.ModuleList(ffe_layers)
        self.compression_layers = None
        if feat_compression is not None:
            self.compression_layers = nn.Sequential(
                nn.Linear(feat_channels[-1], feat_compression),
                nn.ReLU(inplace=True))

    def forward(self, voxel_dict: dict) -> dict:
        features = voxel_dict['voxels']
        coors = voxel_dict['coors'][:, :3] # only use batch, x, y
        batch_size = coors[:, 0].max().item() + 1

        # 2D Voxels
        voxel_coors, inverse_map = torch.unique(
            coors, return_inverse=True, dim=0)
        voxel_coors = voxel_coors.to(torch.int64)
        
        if self.pre_norm is not None:
            features = self.pre_norm(features)

        point_feats = []
        for ffe in self.ffe_layers:
            features = ffe(features)
            point_feats.append(features)
        voxel_feats = torch_scatter.scatter_max(
            features, inverse_map, dim=0)[0]

        if self.compression_layers is not None:
            voxel_feats = self.compression_layers(voxel_feats)
        
        # get bev feature map
        bev_feats_map = torch.zeros(batch_size, self.grid_shape[0], self.grid_shape[1], voxel_feats.shape[-1], device=voxel_feats.device, dtype=voxel_feats.dtype)
        bev_feats_map[voxel_coors[:, 0], voxel_coors[:, 1], voxel_coors[:, 2]] = voxel_feats
        bev_feats_map = bev_feats_map.permute(0, 3, 1, 2)
        
        voxel_dict['bev_feats_map'] = bev_feats_map
        voxel_dict['voxel_feats'] = voxel_feats
        voxel_dict['voxel_coors'] = voxel_coors
        voxel_dict['point_feats'] = point_feats

        return voxel_dict