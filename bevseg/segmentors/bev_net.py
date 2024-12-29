from typing import Dict, List

from torch import Tensor

from mmdet3d.models import EncoderDecoder3D
from mmdet3d.registry import MODELS
from mmdet3d.structures import PointData
from mmdet3d.structures.det3d_data_sample import OptSampleList, SampleList
from mmdet3d.utils import ConfigType, OptConfigType, OptMultiConfig
from torch import Tensor

@MODELS.register_module()
class BevNet(EncoderDecoder3D):
    def __init__(self,
                 bev_encoder: ConfigType,
                 backbone: ConfigType,
                 decode_head: ConfigType,
                 neck: OptConfigType = None,
                 auxiliary_head: OptMultiConfig = None,
                 train_cfg: OptConfigType = None,
                 test_cfg: OptConfigType = None,
                 data_preprocessor: OptConfigType = None,
                 init_cfg: OptMultiConfig = None) -> None:
        super(BevNet, self).__init__(
            backbone=backbone,
            decode_head=decode_head,
            neck=neck,
            auxiliary_head=auxiliary_head,
            train_cfg=train_cfg,
            test_cfg=test_cfg,
            data_preprocessor=data_preprocessor,
            init_cfg=init_cfg)
        self.bev_encoder = MODELS.build(bev_encoder)
        
        
    def extract_feat(self, batch_inputs_dict: dict) -> dict:
        voxel_dict = batch_inputs_dict['voxels']
        voxel_dict = self.bev_encoder(voxel_dict)
        voxel_dict = self.backbone(voxel_dict)
        if self.with_neck:
            voxel_dict = self.neck(voxel_dict)
        return voxel_dict
    
    
    def loss(self, 
             batch_inputs_dict: dict,
             batch_data_samples: SampleList) -> Dict[str, Tensor]:
        voxel_dict = self.extract_feat(batch_inputs_dict)
        losses = dict()
        loss_decode = self._decode_head_forward_train(voxel_dict,
                                                      batch_data_samples)
        losses.update(loss_decode)

        if self.with_auxiliary_head:
            loss_aux = self._auxiliary_head_forward_train(
                voxel_dict, batch_data_samples)
            losses.update(loss_aux)
        return losses
    
    def postprocess_result(self, 
                           seg_labels_list: List[Tensor],
                           batch_data_samples: SampleList) -> SampleList:
        for i, seg_pred in enumerate(seg_labels_list):
            batch_data_samples[i].set_data(
                {'pred_pts_seg': PointData(**{'pts_semantic_mask': seg_pred})})
        return batch_data_samples


    def predict(self,
            batch_inputs_dict: dict,
            batch_data_samples: SampleList,
            rescale: bool = True) -> SampleList:
        batch_input_metas = []
        for data_sample in batch_data_samples:
            batch_input_metas.append(data_sample.metainfo)
        voxel_dict = self.extract_feat(batch_inputs_dict)
        seg_pred_list = self.decode_head.predict(voxel_dict, batch_input_metas, self.test_cfg)
        return self.postprocess_result(seg_pred_list, batch_data_samples)
        
    
        
        
        
