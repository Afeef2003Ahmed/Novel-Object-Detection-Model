import torch
import torchvision
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn.functional as F
import time
from torchvision import datasets,transforms
import os
from PIL import Image
from torchvision.datasets import VOCDetection
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.utils.data._utils.collate import default_collate
import torch.cuda.amp as amp
import math
from Utilities import generate_anchors,dist2bbox

''' *********************************** Backbone ****************************************************************** '''

def split_feature_map(feature_map):
    channels = feature_map.size(1)
    split_size = channels // 2
    return feature_map[:, :split_size, :, :], feature_map[:, split_size:, :, :]

def merge_feature_map(part1, part2):
    return torch.cat((part1, part2), dim=1)

class Stem(nn.Module):
    def __init__(self, in_channels):
        super(Stem, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 24, 3, stride=2, padding=1, bias=False),
            nn.Conv2d(24, 24, 3, stride=1, padding=1, bias=False),
            nn.Conv2d(24, 48, 3, stride=1, padding=1, bias=False),
            nn.MaxPool2d(3, stride=2, padding=1),
            nn.Conv2d(48, 64, 1, stride=1, padding=0, bias=False),
            nn.Conv2d(64, 128, 3, stride=1, padding=1, bias=False),
            nn.MaxPool2d(3, stride=2, padding=1),
        )  # (Batch_size, 128, 52, 52)
        
        self.branch_0 = nn.Conv2d(128, 64, 1, stride=1, padding=0, bias=False) # (Batch_size, 64, 52, 52)
        self.branch_1 = nn.Sequential(
            nn.Conv2d(128, 32, 1, stride=1, padding=0, bias=False),
            nn.Conv2d(32, 48, 5, stride=1, padding=2, bias=False),
        ) # (Batch_size, 48, 52, 52)
        self.branch_2 = nn.Sequential(
            nn.Conv2d(128, 48, 1, stride=1, padding=0, bias=False),
            nn.Conv2d(48, 64, 3, stride=1, padding=1, bias=False),
            nn.Conv2d(64, 64, 3, stride=1, padding=1, bias=False),
        ) # (Batch_size, 64, 52, 52)
        self.branch_3 = nn.Sequential(
            nn.AvgPool2d(3, stride=1, padding=1, count_include_pad=False),
            nn.Conv2d(128, 48, 1, stride=1, padding=0, bias=False)
        ) # (Batch_size, 48, 52, 52)

    def forward(self, x):
        x = self.features(x)
        x0 = self.branch_0(x)
        x1 = self.branch_1(x)
        x2 = self.branch_2(x)
        x3 = self.branch_3(x)
        
        '''print(f'' **** Stem shapes ****
              Features shape: {x.shape}
              Branch 0 shape: {x0.shape}
              Branch 1 shape: {x1.shape}
              Branch 2 shape: {x2.shape}
              Branch 3 shape: {x3.shape}
              Output   shape: {torch.cat((x0,x1,x2,x3),dim=1).shape}'')'''
        
        return torch.cat((x0, x1, x2, x3), dim=1) # Output: (Batch_size, 224, 52, 52)

    

class Block_A(nn.Module):
    def __init__(self, in_channels, scale=1.0):
        super(Block_A, self).__init__()
        self.scale = scale
        self.branch_0 = nn.Conv2d(in_channels//2, 24, 1, stride=1, padding=0, bias=False) # (batch_size, 24, 52, 52)
        self.branch_1 = nn.Sequential(
            nn.Conv2d(in_channels//2, 24, 1, stride=1, padding=0, bias=False),
            nn.Conv2d(24, 24, 3, stride=1, padding=1, bias=False)
        ) # (batch_size, 24, 52, 52)
        self.branch_2 = nn.Sequential(
            nn.Conv2d(in_channels//2, 24, 1, stride=1, padding=0, bias=False),
            nn.Conv2d(24, 32, 3, stride=1, padding=1, bias=False),
            nn.Conv2d(32, 48, 3, stride=1, padding=1, bias=False)
        ) # (batch_size, 48, 52, 52)
        self.conv = nn.Conv2d(96, 112, 1, stride=1, padding=0, bias=True) 
        self.relu = nn.ReLU(inplace=True) # After Residual connection : (batch_size, 112, 52, 52)

    def forward(self, x):
        part1, part2 = split_feature_map(x)
        x0 = self.branch_0(part1)
        x1 = self.branch_1(part1)
        x2 = self.branch_2(part1)
        x_res = torch.cat((x0, x1, x2), dim=1)
        x_res = self.conv(x_res)
        x_res = self.relu(part1 + self.scale * x_res)
        
        '''print(f'' **** Block A shapes ****
              Branch 0 shape: {x0.shape}
              Branch 1 shape: {x1.shape}
              Branch 2 shape: {x2.shape}
              Residual Branch shape: {x_res.shape}
              Output   shape: {torch.cat((part2,x_res),dim=1).shape}'')'''
        
        
        return torch.cat((part2, x_res), dim=1) # (batch_size, 224, 52, 52)
    

class Reduction_A(nn.Module):
    def __init__(self, in_channels, k, l, m, n):
        super(Reduction_A, self).__init__()
        self.branch_0 = nn.Conv2d(in_channels, n, 3, stride=2, padding=1, bias=False) # (batch_size, 256, 26, 26)
        self.branch_1 = nn.Sequential(
            nn.Conv2d(in_channels, k, 1, stride=1, padding=0, bias=False),
            nn.Conv2d(k, l, 3, stride=1, padding=1, bias=False),
            nn.Conv2d(l, m, 3, stride=2, padding=1, bias=False),
        ) # (batch_size, 256, 26, 26)
        self.branch_2 = nn.MaxPool2d(3, stride=2, padding=1) # (batch_size, 224, 26, 26)

    def forward(self, x):
        x0 = self.branch_0(x)
        x1 = self.branch_1(x)
        x2 = self.branch_2(x)
        
        '''print(f'' **** Reduction A shapes ****
              Branch 0 shape: {x0.shape}
              Branch 1 shape: {x1.shape}
              Branch 2 shape: {x2.shape}
              Output   shape: {torch.cat((x0,x1,x2),dim=1).shape}'') '''
         
        return torch.cat((x0, x1, x2), dim=1) # (batch_size, 736, 26, 26)

        
class Block_B(nn.Module):
    def __init__(self, in_channels, scale=1.0):
        super(Block_B, self).__init__()
        self.scale = scale
        half_channels = in_channels // 2
        self.branch_0 = nn.Conv2d(half_channels, 128, 1, stride=1, padding=0, bias=False) # (batch_size, 128, 26, 26)
        self.branch_1 = nn.Sequential(
            nn.Conv2d(half_channels, 96, 1, stride=1, padding=0, bias=False),
            nn.Conv2d(96, 112, (1, 7), stride=1, padding=(0, 3), bias=False),
            nn.Conv2d(112, 128, (7, 1), stride=1, padding=(3, 0), bias=False)
        ) # (batch_size, 128, 26, 26)
        self.conv = nn.Conv2d(256, half_channels, 1, stride=1, padding=0, bias=True)
        self.relu = nn.ReLU(inplace=True) # (batch_size, 368, 26, 26)

    def forward(self, x):
        part1, part2 = split_feature_map(x) # (batch_size, 112 , 52, 52)
        x0 = self.branch_0(part1) 
        x1 = self.branch_1(part1)
        x_res = torch.cat((x0, x1), dim=1)
        x_res = self.conv(x_res)
        x_res = self.relu(part1 + self.scale * x_res)
        
        '''print(f'' **** Block B shapes ****
              Branch 0 shape: {x0.shape}
              Branch 1 shape: {x1.shape}
              Residual shape: {x_res.shape}
              Output   shape: {torch.cat((x_res,part2),dim=1).shape}'')'''
        
        return torch.cat((x_res, part2), dim=1) # (batch_size, 736, 26, 26)
    
    

class Reduction_B(nn.Module):
    def __init__(self, in_channels):
        super(Reduction_B, self).__init__()
        self.branch_0 = nn.Sequential(
            nn.Conv2d(in_channels, 192, 1, stride=1, padding=0, bias=False),
            nn.Conv2d(192, 256, 3, stride=2, padding=1, bias=False)
        ) # (batch_size, 256, 13, 13)
        self.branch_1 = nn.Sequential(
            nn.Conv2d(in_channels, 192, 1, stride=1, padding=0, bias=False),
            nn.Conv2d(192, 224, 3, stride=2, padding=1, bias=False),
        ) # (batch_size, 224, 13, 13)
        self.branch_2 = nn.Sequential(
            nn.Conv2d(in_channels, 192, 1, stride=1, padding=0, bias=False),
            nn.Conv2d(192, 224, 3, stride=1, padding=1, bias=False),
            nn.Conv2d(224, 256, 3, stride=2, padding=1, bias=False)
        ) # (batch_size, 256, 13, 13)
        self.branch_3 = nn.MaxPool2d(3, stride=2, padding=1) # (batch_size, 736, 13, 13)

    def forward(self, x):
        x0 = self.branch_0(x)
        x1 = self.branch_1(x)
        x2 = self.branch_2(x)
        x3 = self.branch_3(x)
        
        '''print(f'' **** Reduction B shapes ****
              Branch 0 shape: {x0.shape}
              Branch 1 shape: {x1.shape}
              Branch 2 shape: {x2.shape}
              Branch 3 shape: {x3.shape}
              Output   shape: {torch.cat((x0,x1,x2,x3),dim=1).shape}'')'''
        
        return torch.cat((x0, x1, x2, x3), dim=1) # (batch_size, 1472, 13, 13)

        
        

class Block_C(nn.Module):
    def __init__(self, in_channels, scale=1.0, activation=True):
        super(Block_C, self).__init__()
        self.scale = scale
        self.activation = activation
        self.branch_0 = nn.Conv2d(in_channels//2, 128, 1, stride=1, padding=0, bias=False)
        self.branch_1 = nn.Sequential(
            nn.Conv2d(in_channels//2, 128, 1, stride=1, padding=0, bias=False),
            nn.Conv2d(128, 160, (1, 3), stride=1, padding=(0, 1), bias=False),
            nn.Conv2d(160, 192, (3, 1), stride=1, padding=(1, 0), bias=False)
        )
        self.conv = nn.Conv2d(320, 736, 1, stride=1, padding=0, bias=True)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        part1, part2 = split_feature_map(x)
        x0 = self.branch_0(part1)
        x1 = self.branch_1(part1)
        x_res = torch.cat((x0, x1), dim=1)
        x_res = self.conv(x_res)
        x_res = self.relu(part1 + self.scale * x_res)
        
        '''print(f'' **** Block C shapes ****
              Branch 0 shape: {x0.shape}
              Branch 1 shape: {x1.shape}
              Residual Branch shape: {x_res.shape}
              Output   shape: {torch.cat((part2,x_res),dim=1).shape}'')'''
        
        return torch.cat((part2, x_res), dim=1)

class backbone(nn.Module):
    def __init__(self, in_channels=3, classes=10, k=192, l=192, m=256, n=256):
        super(backbone, self).__init__()
        blocks = []
        blocks.append(Stem(in_channels))
        for _ in range(3):
            blocks.append(Block_A(in_channels=224, scale=0.17))
        blocks.append(Reduction_A(224, k, l, m, n))
        for _ in range(5):
            blocks.append(Block_B(736, 0.10))  # Changed from 768 to 736
        blocks.append(Reduction_B(736))  # Changed from 768 to 736
        for _ in range(3):
            blocks.append(Block_C(1472, 0.20))  # Changed from 1536 to 1504
        blocks.append(Block_C(1472, activation=False))  # Changed from 1536 to 1504
        self.features = nn.Sequential(*blocks)

    def forward(self, x):
        outputs = []
        for idx, block in enumerate(self.features):
            x = block(x)
            if idx in [3, 9, 14]:
                outputs.append(x)
        return outputs
    
''' *********************************** Neck ********************************************************* '''

class Neck(nn.Module):
    def __init__(self):
        super(Neck, self).__init__()
        
        # Refinement convolutions
        self.refine1 = nn.Sequential(
            nn.Conv2d(4128, 1032, 1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(1032),
            nn.ReLU(inplace=True),
            nn.Conv2d(1032, 256, 1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )
        
        self.refine2 = nn.Sequential(
            nn.Conv2d(7808, 2808, 1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(2808),
            nn.ReLU(inplace=True),
            nn.Conv2d(2808, 512, 1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True)
        )
        
        self.refine3 = nn.Sequential(
            nn.Conv2d(4640, 2320, 1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(2320),
            nn.ReLU(inplace=True),
            nn.Conv2d(2320, 1024, 1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(1024),
            nn.ReLU(inplace=True)
        )
        
        # Additional feature fusion layers
        self.fusion1 = nn.Conv2d(256, 256, 3, padding=1)
        self.fusion2 = nn.Conv2d(512, 512, 3, padding=1)
        self.fusion3 = nn.Conv2d(1024, 1024, 3, padding=1)

    def forward(self, scale_preds):
        self.scale_1 = scale_preds[0]
        self.scale_2 = scale_preds[1]
        self.scale_3 = scale_preds[2]
        
        #print(f''' Scale Inputs
        #     scale 1 shape: {self.scale_1.shape}
        #     scale 2 shape: {self.scale_2.shape}
        #     scale 3 shape: {self.scale_3.shape}''') 
                     

        # Downsampling and upsampling
        scale_2_d = self.downsample(self.scale_2, (self.scale_1.size(2), self.scale_1.size(3)))
        out1 = torch.cat((scale_2_d, self.scale_1), dim=1)
        out_1_up = self.upsample(out1, (self.scale_2.size(2), self.scale_2.size(3)))
        scale_3_d = self.downsample(self.scale_3, (self.scale_2.size(2), self.scale_2.size(3)))
        out_2 = torch.cat((scale_3_d, self.scale_2), dim=1)
        out_2 = torch.cat((out_1_up, out_2), dim=1)
        out_2_d = self.downsample(out_2, (out1.size(2), out1.size(3)))
        scale1_output = torch.cat((out_2_d, out1), dim=1)
        
        # Refinement and fusion for scale 1
        #print(f"scale1_output {scale1_output.shape}")
        output_scale1 = self.refine1(scale1_output)
        output_scale1 = output_scale1 + self.fusion1(output_scale1)  # Skip connection
        
        out2_up = self.upsample(out_2, (self.scale_3.size(2), self.scale_3.size(3)))
        out3 = torch.cat((out2_up, self.scale_3), dim=1)
        out3_d = self.downsample(out3, (out_2.size(2), out_2.size(3)))
        scale2_output = torch.cat((out3_d, out_2), dim=1)
        #print(f"scale2_output {scale2_output.shape}")
        # Refinement and fusion for scale 2
        output_scale2 = self.refine2(scale2_output)
        output_scale2 = output_scale2 + self.fusion2(output_scale2)  # Skip connection
        
        scale3_output = out3
        #print(f"scale3_output {scale3_output.shape}")
        # Refinement and fusion for scale 3
        output_scale3 = self.refine3(scale3_output)
        output_scale3 = output_scale3 + self.fusion3(output_scale3)  # Skip connection
        
        outputs = [output_scale1, output_scale2, output_scale3]
        return outputs

    def downsample(self, x, size):
        return F.interpolate(x, size=size, mode='bilinear', align_corners=False)

    def upsample(self, x, size):
        return F.interpolate(x, size=size, mode='bilinear', align_corners=False)


device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu:0')

''' ****************************** Head ********************************************* '''

class Detect(nn.Module):
    export = False

    def __init__(self, num_classes=11, num_layers=3, inplace=True, head_layers=None, use_dfl=False, reg_max=0):
        super().__init__()
        
        assert head_layers is not None
        self.nc = num_classes
        self.no = num_classes + 5
        self.nl = num_layers
        self.grid = [torch.zeros(1)] * num_layers
        self.prior_prob = 1e-2
        self.inplace = inplace
        stride = [8, 16, 32]
        self.stride = torch.tensor(stride, device=device)
        self.use_dfl = use_dfl
        self.reg_max = reg_max
        self.grid_cell_offset = 0.5
        self.grid_cell_size = 5.0
        self.proj_conv = nn.Conv2d(self.reg_max + 1, 1, 1, bias=False)
        

        self.stems = nn.ModuleList()
        self.cls_convs = nn.ModuleList()
        self.reg_convs = nn.ModuleList()
        self.cls_preds = nn.ModuleList()
        self.reg_preds = nn.ModuleList()

        # Efficient decoupled head layers
        for i in range(num_layers):
            idx = i * 5
            self.stems.append(head_layers[idx])
            self.cls_convs.append(head_layers[idx + 1])
            self.reg_convs.append(head_layers[idx + 2])
            self.cls_preds.append(head_layers[idx + 3])
            self.reg_preds.append(head_layers[idx + 4])

        self.initialize_biases()

    def initialize_biases(self):
        for conv in self.cls_preds:
            nn.init.constant_(conv.bias, -math.log((1 - self.prior_prob) / self.prior_prob))
            nn.init.zeros_(conv.weight)

        for conv in self.reg_preds:
            nn.init.constant_(conv.bias, 1.0)
            nn.init.zeros_(conv.weight)

        self.proj = nn.Parameter(torch.linspace(0, self.reg_max, self.reg_max + 1).to(device), requires_grad=False)
        self.proj_conv.weight = nn.Parameter(self.proj.view([1, self.reg_max + 1, 1, 1]).clone().detach().to(device),
                                              requires_grad=False)

    @torch.cuda.amp.autocast()
    def forward(self, x):
        if self.training:
            cls_score_list = []
            reg_distri_list = []

            for i in range(self.nl):
                x[i] = self.stems[i](x[i])
                cls_x = x[i]
                reg_x = x[i]
                cls_feat = self.cls_convs[i](cls_x)
                cls_output = self.cls_preds[i](cls_feat)
                reg_feat = self.reg_convs[i](reg_x)
                reg_output = self.reg_preds[i](reg_feat)

                cls_output = torch.sigmoid(cls_output)
                cls_score_list.append(cls_output.flatten(2).permute((0, 2, 1)))
                reg_distri_list.append(reg_output.flatten(2).permute((0, 2, 1)))

            cls_score_list = torch.cat(cls_score_list, dim=1)
            reg_distri_list = torch.cat(reg_distri_list, dim=1)

            return x, cls_score_list, reg_distri_list
        else:
            cls_score_list = []
            reg_dist_list = []

            for i in range(self.nl):
                b, _, h, w = x[i].shape
                #print(x[i].shape)
                l = h * w
                x[i] = self.stems[i](x[i])
                cls_x = x[i]
                reg_x = x[i]
                cls_feat = self.cls_convs[i](cls_x)
                cls_output = self.cls_preds[i](cls_feat)
                reg_feat = self.reg_convs[i](reg_x)
                reg_output = self.reg_preds[i](reg_feat)


                if self.use_dfl:
                    reg_output = reg_output.reshape([-1, 4, self.reg_max + 1, l]).permute(0, 2, 1, 3)
                    reg_output = self.proj_conv(F.softmax(reg_output, dim=1))
                    print(f'Reg_output {reg_output.shape}')
                cls_output = torch.sigmoid(cls_output)

                if self.export:
                    cls_score_list.append(cls_output)
                    reg_dist_list.append(reg_output)
                else:
                    cls_score_list.append(cls_output.reshape([b, self.nc, l]))
                    reg_dist_list.append(reg_output.reshape([b, 4, l]))

            if self.export:
                return tuple(torch.cat([cls, reg], dim=1) for cls, reg in zip(cls_score_list, reg_dist_list))

            cls_score_list = torch.cat(cls_score_list, dim=-1).permute(0, 2, 1)
            reg_dist_list = torch.cat(reg_dist_list, dim=-1).permute(0, 2, 1)

            anchor_points, stride_tensor = generate_anchors(
                x, self.stride, self.grid_cell_size, self.grid_cell_offset, device=x[0].device, is_eval=True, mode='af')

            pred_bboxes = dist2bbox(reg_dist_list, anchor_points, box_format='xywh')
            pred_bboxes *= stride_tensor
            return torch.cat(
                [
                    pred_bboxes,
                    torch.ones((b, pred_bboxes.shape[1], 1), device=pred_bboxes.device, dtype=pred_bboxes.dtype),
                    cls_score_list
                ],
                dim=-1)

activation_table = {'relu':nn.ReLU(),
                    'silu':nn.SiLU(),
                    'hardswish':nn.Hardswish()
                    }

class SiLU(nn.Module):
    '''Activation of SiLU'''
    @staticmethod
    def forward(x):
        return x * torch.sigmoid(x)

class ConvModule(nn.Module):
    '''A combination of Conv + BN + Activation'''
    def __init__(self, in_channels, out_channels, kernel_size, stride, activation_type, padding=None, groups=1, bias=False):
        super().__init__()
        if padding is None:
            padding = kernel_size // 2
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            groups=groups,
            bias=bias,
        )
        self.bn = nn.BatchNorm2d(out_channels)
        if activation_type is not None:
            self.act = activation_table.get(activation_type)
        self.activation_type = activation_type

    def forward(self, x):
        if self.activation_type is None:
            return self.bn(self.conv(x))
        return self.act(self.bn(self.conv(x)))

    def forward_fuse(self, x):
        if self.activation_type is None:
            return self.conv(x)
        return self.act(self.conv(x))


class ConvBNSiLU(nn.Module):
    '''Conv and BN with SiLU activation'''
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=None, groups=1, bias=False):
        super().__init__()
        self.block = ConvModule(in_channels, out_channels, kernel_size, stride, 'silu', padding, groups, bias)

    def forward(self, x):
        return self.block(x)


def build_effidehead_layer(channels_list, num_anchors, num_classes, reg_max=0, num_layers=3):

    chx = [0, 1,2]

    head_layers = nn.Sequential(
        # stem0
        ConvBNSiLU(
            in_channels=channels_list[chx[0]],
            out_channels=channels_list[chx[0]],
            kernel_size=1,
            stride=1
        ),
        # cls_conv0
        ConvBNSiLU(
            in_channels=channels_list[chx[0]],
            out_channels=channels_list[chx[0]],
            kernel_size=3,
            stride=1
        ),
        # reg_conv0
        ConvBNSiLU(
            in_channels=channels_list[chx[0]],
            out_channels=channels_list[chx[0]],
            kernel_size=3,
            stride=1
        ),
        # cls_pred0
        nn.Conv2d(
            in_channels=channels_list[chx[0]],
            out_channels=num_classes * num_anchors,
            kernel_size=1
        ),
        # reg_pred0
        nn.Conv2d(
            in_channels=channels_list[chx[0]],
            out_channels=4 * (reg_max + num_anchors),
            kernel_size=1
        ),
        # stem1
        ConvBNSiLU(
            in_channels=channels_list[chx[1]],
            out_channels=channels_list[chx[1]],
            kernel_size=1,
            stride=1
        ),
        # cls_conv1
        ConvBNSiLU(
            in_channels=channels_list[chx[1]],
            out_channels=channels_list[chx[1]],
            kernel_size=3,
            stride=1
        ),
        # reg_conv1
        ConvBNSiLU(
            in_channels=channels_list[chx[1]],
            out_channels=channels_list[chx[1]],
            kernel_size=3,
            stride=1
        ),
        # cls_pred1
        nn.Conv2d(
            in_channels=channels_list[chx[1]],
            out_channels=num_classes * num_anchors,
            kernel_size=1
        ),
        # reg_pred1
        nn.Conv2d(
            in_channels=channels_list[chx[1]],
            out_channels=4 * (reg_max + num_anchors),
            kernel_size=1
        ),
        # stem2
        ConvBNSiLU(
            in_channels=channels_list[chx[2]],
            out_channels=channels_list[chx[2]],
            kernel_size=1,
            stride=1
        ),
        # cls_conv2
        ConvBNSiLU(
            in_channels=channels_list[chx[2]],
            out_channels=channels_list[chx[2]],
            kernel_size=3,
            stride=1
        ),
        # reg_conv2
        ConvBNSiLU(
            in_channels=channels_list[chx[2]],
            out_channels=channels_list[chx[2]],
            kernel_size=3,
            stride=1
        ),
        # cls_pred2
        nn.Conv2d(
            in_channels=channels_list[chx[2]],
            out_channels=num_classes * num_anchors,
            kernel_size=1
        ),
        # reg_pred2
        nn.Conv2d(
            in_channels=channels_list[chx[2]],
            out_channels=4 * (reg_max + num_anchors),
            kernel_size=1
        )
    )



    return head_layers

''' **************************************************** Model *********************************** '''

class Model(nn.Module):
    def __init__(self, num_classes, backbone, channels_list):
        super().__init__()

        self.backbone = backbone
        self.neck = Neck()
        head_layers = build_effidehead_layer(channels_list, 1, num_classes, reg_max=0, num_layers=3)
        self.head = Detect(num_classes, 3, head_layers=head_layers, use_dfl=False)

    def forward(self, x):
        x = self.backbone(x)
        x = self.neck(x)
        #print(f'output of Neck {x[0].shape}')
        x = self.head(x)
        return x
