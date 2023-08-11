# Copyright (c) OpenMMLab. All rights reserved.
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import ConvModule, DepthwiseSeparableConvModule
from mmcv.runner import BaseModule

from ..builder import NECKS
from ..utils import CSPLayer


# 通道注意力机制
class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=4):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc1   = nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2   = nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out)


# 空间注意力机制
class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1

        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)


class IMA(nn.Module):
    def __init__(self, f_channels, z_channels=256):
        super().__init__()
        self.conv1x1_z = nn.Conv2d(z_channels, f_channels, 1, 1, 0)
        self.channelattention = ChannelAttention(in_planes=f_channels, ratio=4)
        self.spatialattention = SpatialAttention(kernel_size=7)

    def forward(self, z, f):
        """
        z: pearlgan的encoded feature. 形状为(B, 256, H/4, W/4)
        f: necks内部的feature map. 形状为(B, f_channels, h, w)
        """
        # nn.AdaptiveAvgPool2d(1)
        z = F.adaptive_avg_pool2d(z, f.shape[-2:])  # (B, 256, h, w)
        F_cross = self.conv1x1_z(z) * f                   # (B, f_channels, h, w)
        channel_branch = self.channelattention(F_cross) # (B, f_channels, 1, 1)
        spatial_branch = self.spatialattention(F_cross) # (B, 1, h, w)
        out = f * channel_branch * spatial_branch + f

        # return out # 20230423为了visualization注释掉
        return out, self.conv1x1_z(z), F_cross  # 20230423为了visualization添加



@NECKS.register_module()
class YOLOXPAFPN_TIR(BaseModule):
    """Path Aggregation Network used in YOLOX.

    Args:
        in_channels (List[int]): Number of input channels per scale.
        out_channels (int): Number of output channels (used at each scale)
        num_csp_blocks (int): Number of bottlenecks in CSPLayer. Default: 3
        use_depthwise (bool): Whether to depthwise separable convolution in
            blocks. Default: False
        upsample_cfg (dict): Config dict for interpolate layer.
            Default: `dict(scale_factor=2, mode='nearest')`
        conv_cfg (dict, optional): Config dict for convolution layer.
            Default: None, which means using conv2d.
        norm_cfg (dict): Config dict for normalization layer.
            Default: dict(type='BN')
        act_cfg (dict): Config dict for activation layer.
            Default: dict(type='Swish')
        init_cfg (dict or list[dict], optional): Initialization config dict.
            Default: None.
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 num_csp_blocks=3,
                 use_depthwise=False,
                 upsample_cfg=dict(scale_factor=2, mode='nearest'),
                 conv_cfg=None,
                 norm_cfg=dict(type='BN', momentum=0.03, eps=0.001),
                 act_cfg=dict(type='Swish'),
                 init_cfg=dict(
                     type='Kaiming',
                     layer='Conv2d',
                     a=math.sqrt(5),
                     distribution='uniform',
                     mode='fan_in',
                     nonlinearity='leaky_relu')):
        super(YOLOXPAFPN_TIR, self).__init__(init_cfg)
        self.in_channels = in_channels
        self.out_channels = out_channels

        conv = DepthwiseSeparableConvModule if use_depthwise else ConvModule

        # build top-down blocks
        self.upsample = nn.Upsample(**upsample_cfg)
        self.reduce_layers = nn.ModuleList()
        self.top_down_blocks = nn.ModuleList()
        for idx in range(len(in_channels) - 1, 0, -1):
            self.reduce_layers.append(
                ConvModule(
                    in_channels[idx],
                    in_channels[idx - 1],
                    1,
                    conv_cfg=conv_cfg,
                    norm_cfg=norm_cfg,
                    act_cfg=act_cfg))
            self.top_down_blocks.append(
                CSPLayer(
                    in_channels[idx - 1] * 2,
                    in_channels[idx - 1],
                    num_blocks=num_csp_blocks,
                    add_identity=False,
                    use_depthwise=use_depthwise,
                    conv_cfg=conv_cfg,
                    norm_cfg=norm_cfg,
                    act_cfg=act_cfg))

        # build bottom-up blocks
        self.downsamples = nn.ModuleList()
        self.bottom_up_blocks = nn.ModuleList()
        for idx in range(len(in_channels) - 1):
            self.downsamples.append(
                conv(
                    in_channels[idx],
                    in_channels[idx],
                    3,
                    stride=2,
                    padding=1,
                    conv_cfg=conv_cfg,
                    norm_cfg=norm_cfg,
                    act_cfg=act_cfg))
            self.bottom_up_blocks.append(
                CSPLayer(
                    in_channels[idx] * 2,
                    in_channels[idx + 1],
                    num_blocks=num_csp_blocks,
                    add_identity=False,
                    use_depthwise=use_depthwise,
                    conv_cfg=conv_cfg,
                    norm_cfg=norm_cfg,
                    act_cfg=act_cfg))

        self.out_convs = nn.ModuleList()
        for i in range(len(in_channels)):
            self.out_convs.append(
                ConvModule(
                    in_channels[i],
                    out_channels,
                    1,
                    conv_cfg=conv_cfg,
                    norm_cfg=norm_cfg,
                    act_cfg=act_cfg))


        self.ima_bottom_up = nn.ModuleList()
        self.ima_top_down = nn.ModuleList()
        for i in range(len(in_channels)):
            self.ima_bottom_up.append(IMA(in_channels[i], 256))
            if i == 0:
                self.ima_top_down.append(IMA(in_channels[i], 256))
            else:
                self.ima_top_down.append(IMA(in_channels[i-1], 256))



    def forward(self, inputs):
        """
        Args:
            inputs (tuple[Tensor]): input features.

        Returns:
            tuple[Tensor]: YOLOXPAFPN features.
        """
        F_in_list = [] # 20230423为了visualization添加, z对应文章里z_cross
        F_out_list = [] # 20230423为了visualization添加, z对应文章里z_cross
        z_cross_list = []  # 20230423为了visualization添加, z对应文章里z_cross
        z_t2v_list = []  # 20230423为了visualization添加, z对应文章里z_cross

        z, inputs = inputs  # added z(z_t2v): (B, 256, H/4, W/4)
        # print(z.shape)
        assert len(inputs) == len(self.in_channels)

        # top-down path

        ##################### added #####################
        for i in range(len(self.in_channels)):
            # inputs[i] = self.ima_bottom_up[i](z, inputs[i])  # 20230423为了visualization注释掉
            
            # print(f'F_in: {inputs[i]}')
            F_in_list.append(inputs[i])  # 20230423为了visualization添加, z对应文章里z_cross
            inputs[i], z_t2v, F_cross = self.ima_bottom_up[i](z, inputs[i])  # 20230423为了visualization添加, z对应文章里z_cross
            z_cross_list.append(F_cross)    # 20230423为了visualization添加, z对应文章里z_cross
            z_t2v_list.append(z_t2v)    # 20230423为了visualization添加, z对应文章里z_cross
            F_out_list.append(inputs[i])  # 20230423为了visualization添加, z对应文章里z_cross
            # print(f'F_out: {inputs[i]}')

        inputs = tuple(inputs)
        
        ##################### added #####################

        inner_outs = [inputs[-1]]
        for idx in range(len(self.in_channels) - 1, 0, -1):
            feat_heigh = inner_outs[0]
            feat_low = inputs[idx - 1]
            feat_heigh = self.reduce_layers[len(self.in_channels) - 1 - idx](
                feat_heigh)
            inner_outs[0] = feat_heigh

            upsample_feat = self.upsample(feat_heigh)

            inner_out = self.top_down_blocks[len(self.in_channels) - 1 - idx](
                torch.cat([upsample_feat, feat_low], 1))
            inner_outs.insert(0, inner_out)

        # ##################### added #####################
        # inner_outs[0]: (B, in_channels[0], H/8, W/8)
        # inner_outs[1]: (B, in_channels[0], H/4, W/4)
        # inner_outs[2]: (B, in_channels[1], H/2, W/2)
        for i in range(len(self.in_channels)):
            # inner_outs[i] = self.ima_top_down[i](z, inner_outs[i])  # 20230423为了visualization注释掉

            inner_outs[i], z_t2v, F_cross = self.ima_top_down[i](z, inner_outs[i])  # 20230423为了visualization添加, z对应文章里z_cross
            # z_cross_list.append(z_cross)    # 20230423为了visualization添加, z对应文章里z_cross

        # print(inner_outs[0].shape)
        # print(inner_outs[1].shape)
        # print(inner_outs[2].shape)
        # exit()
        # ##################### added #####################

        # bottom-up path
        outs = [inner_outs[0]]
        for idx in range(len(self.in_channels) - 1):
            feat_low = outs[-1]
            feat_height = inner_outs[idx + 1]
            downsample_feat = self.downsamples[idx](feat_low)
            out = self.bottom_up_blocks[idx](
                torch.cat([downsample_feat, feat_height], 1))
            outs.append(out)

        # out convs
        for idx, conv in enumerate(self.out_convs):
            outs[idx] = conv(outs[idx])

        # import cv2
        # import numpy as np
        # z = F.avg_pool2d(z, kernel_size=5, stride=1, padding=2)
        # z_vis = torch.mean(z[0], dim=0).detach().cpu().numpy()*255
        # z_vis = cv2.applyColorMap(z_vis.astype(np.uint8), cv2.COLORMAP_JET)
        # cv2.imwrite(f'_work_dir/yolox_tir_m_8x8_300e_coco/visualization/z_vis.jpg', z_vis)
        # for i, z_cross in enumerate(z_cross_list):
        #     print(z_cross.shape, torch.min(z_cross), torch.max(z_cross))
        #     z_cross = F.avg_pool2d(z_cross, kernel_size=5, stride=1, padding=2)
        #     z_cross_vis = torch.mean(z_cross[0], dim=0).detach().cpu().numpy()
        #     print(np.max(z_cross_vis), np.min(z_cross_vis))
        #     z_cross_vis = (z_cross_vis - np.min(z_cross_vis)) / (np.max(z_cross_vis) - np.min(z_cross_vis))*255
        #     z_cross_vis = cv2.applyColorMap(z_cross_vis.astype(np.uint8), cv2.COLORMAP_JET)
        #     cv2.imwrite(f'_work_dir/yolox_tir_m_8x8_300e_coco/visualization/z_{i}.jpg', z_cross_vis)
        #     # print(z_cross.shape)
        # # exit()
        # input("继续：")
        # import os
        # for i in range(200):
        #     save_dir = f'./_work_dir/yolox_tir_m_8x8_300e_coco/CMA-TSNE/{i}'
        #     if os.exists(save_dir):

        # os.makedirs('./_work_dir/yolox_tir_m_8x8_300e_coco/CMA-TSNE/CMA1', exist_ok=True)
        # os.makedirs('./_work_dir/yolox_tir_m_8x8_300e_coco/CMA-TSNE/CMA2', exist_ok=True)
        # os.makedirs('./_work_dir/yolox_tir_m_8x8_300e_coco/CMA-TSNE/CMA3', exist_ok=True)
        # os.makedirs('./_work_dir/yolox_tir_m_8x8_300e_coco/CMA-TSNE/CMA4', exist_ok=True)
        # os.makedirs('./_work_dir/yolox_tir_m_8x8_300e_coco/CMA-TSNE/CMA5', exist_ok=True)
        # os.makedirs('./_work_dir/yolox_tir_m_8x8_300e_coco/CMA-TSNE/CMA6', exist_ok=True)
        
        # for j in range(1000):
        #     if os.path.exists(f'./_work_dir/yolox_tir_m_8x8_300e_coco/CMA-TSNE/{j}'):
        #         continue
        #     else:
        #         print(f'第{j}张图')
        #         for i, z_cross in enumerate(z_cross_list):
        #             print(i, z_cross.shape, torch.min(z_cross), torch.max(z_cross))
        #             print(i, F_in_list[i].shape)
        #             print(i, F_out_list[i].shape)
        #             print(i, z_t2v_list[i].shape)

        #             # for j in range(1000):
        #             #     if os.path.exists(f'./_work_dir/yolox_tir_m_8x8_300e_coco/CMA-TSNE/{j}'):
        #             #         continue
        #             #     else:
        #             #         print(f'j: {j}')
        #             os.makedirs(f'./_work_dir/yolox_tir_m_8x8_300e_coco/CMA-TSNE/{j}/CMA{i+1}', exist_ok=True)
        #             torch.save(z_cross.cpu(), f'./_work_dir/yolox_tir_m_8x8_300e_coco/CMA-TSNE/{j}/CMA{i+1}/F_cross.pth')
        #             torch.save(F_in_list[i].cpu(), f'./_work_dir/yolox_tir_m_8x8_300e_coco/CMA-TSNE/{j}/CMA{i+1}/F_in.pth')
        #             torch.save(F_out_list[i].cpu(), f'./_work_dir/yolox_tir_m_8x8_300e_coco/CMA-TSNE/{j}/CMA{i+1}/F_out.pth')
        #             torch.save(z_t2v_list[i].cpu(), f'./_work_dir/yolox_tir_m_8x8_300e_coco/CMA-TSNE/{j}/CMA{i+1}/z_t2v.pth')
        #             # torch.save(F.adaptive_avg_pool2d(z.cpu(), output_size=(8, 8)), f'./_work_dir/yolox_tir_m_8x8_300e_coco/CMA-TSNE/{j}/CMA{i+1}/z_t2v.pth')
                
        #         break
        

        # # input("继续：")



        return tuple(outs)
