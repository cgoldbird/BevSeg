from typing import List, Tuple

import torch
from torch import Tensor, nn

from mmdet3d.models import Base3DDecodeHead
from mmdet3d.registry import MODELS
from mmdet3d.structures.det3d_data_sample import SampleList
from mmdet3d.utils import ConfigType, OptConfigType, OptMultiConfig

def reshape_to_voxel(x: Tensor, n_height: int, n_class: int) -> Tensor:
    x = x.permute(0,2,3,1)
    voxel_shape =  [x.size()[0], x.size()[1], x.size()[2], n_height, n_class]
    x = x.view(voxel_shape).permute(0,4,1,2,3)
    return x

@MODELS.register_module()
class BevHead(Base3DDecodeHead):
    def __init__(self,
                 channels: int,
                 n_height: int,
                 num_classes: int,
                 dropout_ratio: float = 0,
                 loss_ce: ConfigType = dict(
                     type='mmdet.CrossEntropyLoss',
                     use_sigmoid=False,
                     class_weight=None,
                     loss_weight=1.0),
                 loss_lovasz: ConfigType = dict(
                     type='LovaszLoss', loss_weight=1.0, reduction='none'),
                 conv_seg_kernel_size: int = 1,
                 ignore_index: int = 19,
                 init_cfg: OptMultiConfig = None) -> None:
        super(BevHead, self).__init__(
            channels=channels, 
            num_classes=num_classes,
            dropout_ratio=dropout_ratio,
            ignore_index=ignore_index,
            init_cfg=init_cfg,
            conv_seg_kernel_size=conv_seg_kernel_size)
        
        self.loss_ce = MODELS.build(loss_ce)
        if loss_lovasz is not None:
            self.loss_lovasz = MODELS.build(loss_lovasz)
        else:
            self.loss_lovasz = None
        self.ignore_index = ignore_index
        self.n_height = n_height
        # 重写conv_seg
        self.conv_seg = nn.Conv2d(channels, num_classes*self.n_height, kernel_size=conv_seg_kernel_size)

    
    def forward(self, voxel_dict: dict) -> dict:
        seg_logit = self.cls_seg(voxel_dict['bev_feats_map'])
        seg_logit = reshape_to_voxel(seg_logit, self.n_height, self.num_classes)
        voxel_dict['seg_logit'] = seg_logit
        return voxel_dict
        
    
    def _stack_batch_gt(self, batch_data_samples: SampleList) -> Tensor:
        gt_semantic_segs = [
            data_sample.gt_pts_seg.semantic_seg
            for data_sample in batch_data_samples
        ]
        return torch.stack(gt_semantic_segs, dim=0)
    
    
    def loss_by_feat(self, voxel_dict: dict,
                     batch_data_samples: SampleList) -> dict:
        seg_logit = voxel_dict['seg_logit']
        seg_label = self._stack_batch_gt(batch_data_samples)
        
        loss = dict()
        loss['loss_ce'] = self.loss_ce(
            seg_logit, seg_label, ignore_index=self.ignore_index)
        if self.loss_lovasz:
            B, C, L, H, W = seg_logit.size()
            seg_logit_t = seg_logit.contiguous().view(B, C, L, H*W)
            B, C, H, W = seg_logit_t.size()
            seg_logit_t = seg_logit_t.permute(0, 2, 3, 1).contiguous().view(-1, C)
            seg_label_t = seg_label.view(-1)
            loss['loss_lovasz'] = self.loss_lovasz(
                seg_logit_t, seg_label_t, ignore_index=self.ignore_index)
        return loss
    
    
    def predict(self, voxel_dict: dict, batch_input_metas: List[dict],
                test_cfg: ConfigType) -> List[Tensor]:
        voxel_dict = self.forward(voxel_dict)
        # torch.Size([2, 20, 480, 360, 32])
        seg_logit = voxel_dict['seg_logit']
        seg_preds = seg_logit.argmax(dim=1) # voxel-wise label
        coors = voxel_dict['coors']
        seg_preds = seg_preds[coors[:, 0], coors[:, 1], coors[:, 2], coors[:, 3]] # change to points label
        seg_pred_list = []
        for batch_idx in range(len(batch_input_metas)):
            batch_mask = coors[:, 0] == batch_idx
            seg_pred = seg_preds[batch_mask]
            seg_pred_list.append(seg_pred)
        return seg_pred_list
        
            
        

if __name__ == '__main__':
    bev_head = BevHead(channels=64, n_height=32, num_classes=20)
    voxel_dict = {'bev_feats_map': torch.randn(2, 64, 480, 360)}
    out_feats = bev_head(voxel_dict)
    print(out_feats['seg_logit'].shape)
    # torch.Size([2, 20, 480, 360, 32])