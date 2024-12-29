from typing import Optional, Sequence, Tuple

import torch
from mmcv.cnn import (ConvModule, build_activation_layer, build_conv_layer, MaxPool2d,
                      build_norm_layer)
from mmengine.model import BaseModule
from torch import Tensor, nn
from torch.nn import functional as F

from mmdet3d.registry import MODELS
from mmdet3d.utils import ConfigType, OptConfigType, OptMultiConfig
from mmdet.models.layers.dropblock import DropBlock


class CBR(BaseModule):
    def __init__(self, 
                 in_ch: int, 
                 out_ch: int, 
                 double_version: bool = False, 
                 kernel_size: int = 3,
                 stride: int = 1, 
                 padding: int = 1,
                 conv_cfg: OptConfigType = None,
                 norm_cfg: ConfigType = dict(type='BN'),
                 act_cfg: ConfigType = dict(type='LeakyReLU', inplace=True),
                 init_cfg: OptMultiConfig = None) -> None:
        super(CBR, self).__init__(init_cfg)
        
        self.double_version = double_version
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        
        # Define conv1
        self.conv1 = build_conv_layer(conv_cfg, in_ch, out_ch, kernel_size=self.kernel_size, stride=self.stride, padding=self.padding)
        self.norm1_name, norm1 = build_norm_layer(norm_cfg, out_ch, postfix=1)
        self.add_module(self.norm1_name, norm1)
        self.relu1 = build_activation_layer(act_cfg)
        
        # Define conv2
        self.conv2 = build_conv_layer(conv_cfg, out_ch, out_ch, kernel_size=self.kernel_size, stride=self.stride, padding=self.padding)
        self.norm2_name, norm2 = build_norm_layer(norm_cfg, out_ch, postfix=2)
        self.add_module(self.norm2_name, norm2)
        self.relu2 = build_activation_layer(act_cfg)

    @property
    def norm1(self) -> nn.Module:
        """nn.Module: normalization layer after the first convolution layer."""
        return getattr(self, self.norm1_name)

    @property
    def norm2(self) -> nn.Module:
        """nn.Module: normalization layer after the second convolution layer."""
        return getattr(self, self.norm2_name)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.relu1(x)
        
        if self.double_version:
            x = self.conv2(x)
            x = self.norm2(x)
            x = self.relu2(x)
        
        return x


class PolarCBR(BaseModule):
    def __init__(self,
                 in_ch: int,
                 out_ch: int,
                 double_version: bool = False,
                 kernel_size: int = 3,
                 stride: int = 1,
                 padding: int = 1,
                 conv_cfg: OptConfigType = None,
                 norm_cfg: ConfigType = dict(type='BN'),
                 act_cfg: ConfigType = dict(type='LeakyReLU', inplace=True),
                 init_cfg: OptMultiConfig = None) -> None:
        super(PolarCBR, self).__init__(init_cfg)
        
        self.double_version = double_version
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

        # Define conv1
        self.conv1 = build_conv_layer(conv_cfg, in_ch, out_ch, kernel_size=self.kernel_size, stride=self.stride, padding=(self.padding, 0))
        self.norm1_name, norm1 = build_norm_layer(norm_cfg, out_ch, postfix=1)
        self.add_module(self.norm1_name, norm1)
        self.relu1 = build_activation_layer(act_cfg)
        
        # Define conv2
        self.conv2 = build_conv_layer(conv_cfg, out_ch, out_ch, kernel_size=self.kernel_size, stride=self.stride, padding=(self.padding, 0))
        self.norm2_name, norm2 = build_norm_layer(norm_cfg, out_ch, postfix=2)
        self.add_module(self.norm2_name, norm2)
        self.relu2 = build_activation_layer(act_cfg)

    @property
    def norm1(self) -> nn.Module:
        """nn.Module: normalization layer after the first convolution layer."""
        return getattr(self, self.norm1_name)

    @property
    def norm2(self) -> nn.Module:
        """nn.Module: normalization layer after the second convolution layer."""
        return getattr(self, self.norm2_name)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # CITATION: circular padding from https://github.com/edwardzhou130/PolarSeg
        x = F.pad(x, (self.padding, self.padding, 0, 0), mode='circular')
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.relu1(x)
        
        if self.double_version:
            x = F.pad(x, (self.padding, self.padding, 0, 0), mode='circular')
            x = self.conv2(x)
            x = self.norm2(x)
            x = self.relu2(x)
        
        return x
    
    
class up_CBR(nn.Module):
    def __init__(self, 
                 in_ch: int, out_ch: int,
                 basic_block: nn.Module,
                 double_version: bool = True,
                 conv_cfg: OptConfigType = None,
                 norm_cfg: ConfigType = dict(type='BN'),
                 act_cfg: ConfigType = dict(type='LeakyReLU', inplace=True)):
        super(up_CBR, self).__init__()
        self.double_version = double_version
        self.up = nn.ConvTranspose2d(in_ch//2, in_ch//2, 2, stride=2, groups=in_ch//2)
        self.drop  = DropBlock(drop_prob=0.5, block_size=7)
        self.conv = basic_block(in_ch, out_ch, double_version=double_version, conv_cfg=conv_cfg, norm_cfg=norm_cfg, act_cfg=act_cfg)


    def forward(self, x1, x2):
        x1 = self.up(x1)
        
        up_error_x, up_error_y = x2.size()[3] - x1.size()[3], x2.size()[2] - x1.size()[2]
        left_padding, right_padding = up_error_x//2, up_error_x - up_error_x//2
        up_padding, down_padding = up_error_y//2, up_error_y - up_error_y//2
        
        x1 = F.pad(x1, (left_padding, right_padding, up_padding, down_padding))

        x = torch.cat([x2, x1], dim=1)
        x = self.conv(x)

        if self.double_version:
            x = self.drop(x)
        return x


class down_CBR(nn.Module):
    def __init__(self, 
                 in_ch: int, out_ch: int,
                 basic_block: nn.Module,
                 double_version: bool = True,
                 conv_cfg: OptConfigType = None,
                 norm_cfg: ConfigType = dict(type='BN'),
                 act_cfg: ConfigType = dict(type='LeakyReLU', inplace=True)):
        super(down_CBR, self).__init__()
        self.pooling = nn.MaxPool2d(2)
        self.conv = basic_block(in_ch, out_ch, double_version=double_version, conv_cfg=conv_cfg, norm_cfg=norm_cfg, act_cfg=act_cfg)

    def forward(self, x):
        x = self.pooling(x)
        x = self.conv(x)
        return x

@MODELS.register_module()
class Unet(BaseModule):
    def __init__(self, 
                 in_ch: int = 32,
                 inc_ch: int = 64,
                 down_chs: Sequence[int] = (128, 256, 512, 512),
                 up_chs: Sequence[int] = (256, 128, 64, 64),
                 circular_padding: bool = False,
                 double_version: bool = True,
                 conv_cfg: OptConfigType = None,
                 norm_cfg: ConfigType = dict(type='BN'),
                 act_cfg: ConfigType = dict(type='LeakyReLU', inplace=True),
                 init_cfg: OptMultiConfig = None) -> None:
        super(Unet, self).__init__(init_cfg)
        assert len(down_chs) == len(up_chs)
        
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.act_cfg = act_cfg
        self.double_version = double_version
        
        if circular_padding:
            self.basic_block = PolarCBR
        else:
            self.basic_block = CBR
        
        self._make_inc_layer(in_ch, inc_ch)
        self.down_layers = []
        self.up_layers = []
        
        for i, down_ch in enumerate(down_chs):
            dwon_block = down_CBR(inc_ch, down_ch, self.basic_block, double_version=self.double_version, conv_cfg=self.conv_cfg, norm_cfg=self.norm_cfg, act_cfg=self.act_cfg)
            layer_name = f'down{i + 1}'
            self.add_module(layer_name, dwon_block)
            inc_ch = down_ch
            self.down_layers.append(layer_name)
        
        for i, up_ch in enumerate(up_chs):
            up_block = up_CBR(inc_ch*2, up_ch, self.basic_block, double_version=self.double_version, conv_cfg=self.conv_cfg, norm_cfg=self.norm_cfg, act_cfg=self.act_cfg)
            layer_name = f'up{i + 1}'
            self.add_module(layer_name, up_block)
            inc_ch = up_ch
            self.up_layers.append(layer_name)
            
    
    def _make_inc_layer(self, in_ch: int, out_ch: int) -> None:
        self.inc = nn.Sequential(
            build_norm_layer(self.norm_cfg, in_ch)[1],
            self.basic_block(in_ch, out_ch, double_version=self.double_version, conv_cfg=self.conv_cfg, norm_cfg=self.norm_cfg, act_cfg=self.act_cfg)
        )
     

    def forward(self, voxel_dict: dict) -> dict:
        x = voxel_dict['bev_feats_map']
        x = self.inc(x)
        outs = [x]
        for layer_name in self.down_layers:
            down_layer = getattr(self, layer_name)
            x = down_layer(x)
            outs.append(x)
        
        for layer_name, out in zip(self.up_layers, outs[:-1][::-1]):
            up_layer = getattr(self, layer_name)
            x = up_layer(x, out)
        voxel_dict['bev_feats_map'] = x
        return voxel_dict

if __name__ == '__main__':
    backbone = Unet()
    voxel_dict = {'bev_feats_map': torch.randn(2, 64, 480, 360)}
    out_feats = backbone(voxel_dict)
    print(out_feats['bev_feats_map'].shape)
    # torch.Size([2, 64, 480, 360])