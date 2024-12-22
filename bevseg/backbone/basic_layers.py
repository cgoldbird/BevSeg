from typing import Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmdet.models.layers.dropblock import DropBlock

class CBR(nn.Module):
    def __init__(self, 
                 in_ch: int, out_ch: int, 
                 double_version: bool = False, 
                 kernel_size: int = 3,
                 stride: int = 1, padding = 1):
        super(CBR, self).__init__()
        self.double_version = double_version
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, self.kernel_size, self.stride, self.padding),
            nn.BatchNorm2d(out_ch),
            nn.LeakyReLU(inplace=True)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(out_ch, out_ch, self.kernel_size, self.stride, self.padding),
            nn.BatchNorm2d(out_ch),
            nn.LeakyReLU(inplace=True)
        )

    def forward(self, x):
        x = self.conv1(x)
        if self.double_version:
            x = self.conv2(x)
        return x


class polar_CBR(nn.Module):
    def __init__(self,
                 in_ch: int, out_ch: int, 
                 double_version: bool = False, 
                 kernel_size: int = 3, 
                 stride: int = 1, 
                 padding: int = 1):
        super(polar_CBR, self).__init__()
        self.double_version = double_version
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, self.kernel_size, self.stride, (self.padding, 0)),
            nn.BatchNorm2d(out_ch),
            nn.LeakyReLU(inplace=True)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(out_ch, out_ch, self.kernel_size, self.stride, (self.padding, 0)),
            nn.BatchNorm2d(out_ch),
            nn.LeakyReLU(inplace=True)
        )

    def forward(self, x):
        # CITATION: circular padding from https://github.com/edwardzhou130/PolarSeg
        x = F.pad(x,(self.padding, self.padding, 0, 0), mode = 'circular')
        x = self.conv1(x)
        if self.double_version :
            x = F.pad(x,(self.padding, self.padding, 0, 0), mode = 'circular')
            x = self.conv2(x)
        return x

if __name__ == '__main__':
    in_ch = 3
    out_ch = 64
    double_version = True
    kernel_size = 3
    stride = 1
    padding = 1
    cbr = polar_CBR(in_ch, out_ch, double_version, kernel_size, stride, padding)
    print(cbr)
    
class inconv(nn.Module):
    def __init__(self, 
                 in_ch: int, out_ch: int,
                 circular_padding: bool,
                 double_version: bool = False):
        super(inconv, self).__init__()
        self.double_version = double_version

        if circular_padding:
            self.conv = nn.Sequential(
                nn.BatchNorm2d(in_ch),
                polar_CBR(in_ch, out_ch, self.double_version)
            )

        else:
            self.conv = nn.Sequential(
                nn.BatchNorm2d(in_ch),
                CBR(in_ch, out_ch, self.double_version)
            )
        
    def forward(self, x):
        x = self.conv(x)
        return x

class up_CBR(nn.Module):
    def __init__(self, 
                 in_ch: int, out_ch: int,
                 circular_padding: bool,
                 double_version: bool = False):
        super(up_CBR, self).__init__()
        self.circular_padding = circular_padding
        self.double_version = double_version
        self.up = nn.ConvTranspose2d(in_ch//2, in_ch//2, 2, stride=2, groups=in_ch//2)
        self.drop  = DropBlock(drop_prob=0.5, block_size=7)

        if circular_padding :
            self.conv = polar_CBR(in_ch, out_ch, self.double_version)
        else :
            self.conv = CBR(in_ch, out_ch, self.double_version)


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
                 circular_padding: bool, 
                 double_version: bool = False):
        super(down_CBR, self).__init__()
        self.circular_padding = circular_padding
        self.double_version = double_version
        self.pooling = nn.MaxPool2d(2)
        if circular_padding :
            self.conv = polar_CBR(in_ch, out_ch, self.double_version)
        else:
            self.conv = CBR(in_ch, out_ch,self.double_version)

    def forward(self, x):
        x = self.pooling(x)
        x = self.conv(x)
        return x

def reshape_to_voxel(x, n_height, n_class):
    x = x.permute(0,2,3,1)
    voxel_shape =  [x.size()[0], x.size()[1], x.size()[2], n_height, n_class]
    x = x.view(voxel_shape).permute(0,4,1,2,3)
    return x