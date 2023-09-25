# Copyright (c) OpenMMLab. All rights reserved.
import math

import torch
import torch.nn as nn
from mmcv.cnn import ConvModule, DepthwiseSeparableConvModule
from mmcv.runner import BaseModule
from torch.nn.modules.batchnorm import _BatchNorm

from ..builder import BACKBONES
from ..utils import CSPLayer

################################################### PEARLGAN ###################################################
import os
import functools
import torch.nn.functional as F

def get_norm_layer(norm_type='instance'):
    if norm_type == 'batch':
        return functools.partial(nn.BatchNorm2d, affine=True)
    elif norm_type == 'instance':
        return functools.partial(nn.InstanceNorm2d, affine=False)
    else:
        raise NotImplementedError('normalization layer [%s] is not found' % norm_type)


class PGAResBlockv4k3(nn.Module):
    def __init__(self, in_dim, norm_layer, use_bias):
        super(PGAResBlockv4k3, self).__init__()

        self.width = in_dim // 4
        self.bottlenec1 = nn.Sequential(nn.ReflectionPad2d(1), nn.Conv2d(self.width, 1, kernel_size=3, padding=0, bias=use_bias), norm_layer(1), nn.Sigmoid())
        self.bottlenec2 = nn.Sequential(nn.ReflectionPad2d(1), nn.Conv2d(self.width, 1, kernel_size=3, padding=0, bias=use_bias), norm_layer(1), nn.Sigmoid())
        self.bottlenec3 = nn.Sequential(nn.ReflectionPad2d(1), nn.Conv2d(self.width, 1, kernel_size=3, padding=0, bias=use_bias), norm_layer(1), nn.Sigmoid())
        self.bottlenec4 = nn.Sequential(nn.ReflectionPad2d(1), nn.Conv2d(self.width, 1, kernel_size=3, padding=0, bias=use_bias), norm_layer(1), nn.Sigmoid())

        self.ds1 = nn.AvgPool2d(kernel_size=3, stride=2)
        self.ds2 = nn.AvgPool2d(kernel_size=3, stride=4)
        self.ds3 = nn.AvgPool2d(kernel_size=3, stride=8)
        self.ds4 = nn.AvgPool2d(kernel_size=3, stride=16)

        self.conv = nn.Sequential(nn.Conv2d(in_dim, in_dim, kernel_size=1, bias=use_bias), norm_layer(in_dim), nn.PReLU())

        self.softmax  = nn.Softmax(dim=-1) #

    def forward(self, input):
        b, c, h, w = input.size()

        input_fea = self.conv(input)
        spx = torch.split(input_fea, self.width, 1)
        fea_ds1 = self.ds1(spx[0])
        fea_ds2 = self.ds2(spx[1])
        fea_ds3 = self.ds3(spx[2])
        fea_ds4 = self.ds4(spx[3])
    
        # print("fea_ds4:", fea_ds4.shape)
        att_conv1 = self.bottlenec1(fea_ds4)
        att_map1_us = F.interpolate(att_conv1, size=(h, w), mode='bilinear', align_corners=False)
        att_map_g1 = F.interpolate(att_conv1, size=(fea_ds3.size(2), fea_ds3.size(3)), mode='bilinear', align_corners=False)
        
        fea_att1 = att_map_g1.expand_as(fea_ds3).mul(fea_ds3) + fea_ds3
        att_conv2 = self.bottlenec2(fea_att1)
        att_map2_us = F.interpolate(att_conv2, size=(h, w), mode='bilinear', align_corners=False)
        att_map_g2 = F.interpolate(att_conv2, size=(fea_ds2.size(2), fea_ds2.size(3)), mode='bilinear', align_corners=False)

        fea_att2 = att_map_g2.expand_as(fea_ds2).mul(fea_ds2) + fea_ds2
        att_conv3 = self.bottlenec3(fea_att2)
        att_map3_us = F.interpolate(att_conv3, size=(h, w), mode='bilinear', align_corners=False)
        att_map_g3 = F.interpolate(att_conv3, size=(fea_ds1.size(2), fea_ds1.size(3)), mode='bilinear', align_corners=False)
        
        fea_att3 = att_map_g3.expand_as(fea_ds1).mul(fea_ds1) + fea_ds1
        att_conv4 = self.bottlenec4(fea_att3)
        att_map4_us = F.interpolate(att_conv4, size=(h, w), mode='bilinear', align_corners=False)
        
        y1 = att_map4_us.expand_as(spx[0]).mul(spx[0])
        y2 = att_map3_us.expand_as(spx[1]).mul(spx[1])
        y3 = att_map2_us.expand_as(spx[2]).mul(spx[2])
        y4 = att_map1_us.expand_as(spx[3]).mul(spx[3])

        out = torch.cat((y1, y2, y3, y4), 1) + input

        return out, att_map1_us, att_map2_us, att_map3_us, att_map4_us


class SequentialContext(nn.Sequential):
    def __init__(self, n_classes, *args):
        super(SequentialContext, self).__init__(*args)
        self.n_classes = n_classes
        self.context_var = None

    def prepare_context(self, input, domain):
        if self.context_var is None or self.context_var.size()[-2:] != input.size()[-2:]:
            tensor = torch.cuda.FloatTensor if isinstance(input.data, torch.cuda.FloatTensor) \
                     else torch.FloatTensor
            self.context_var = tensor(*((1, self.n_classes) + input.size()[-2:]))

        self.context_var.data.fill_(-1.0)
        self.context_var.data[:,domain,:,:] = 1.0
        return self.context_var

    def forward(self, *input):
        if self.n_classes < 2 or len(input) < 2:
            return super(SequentialContext, self).forward(input[0])
        x, domain = input

        for module in self._modules.values():
            if 'Conv' in module.__class__.__name__:
                context_var = self.prepare_context(x, domain)
                x = torch.cat([x, context_var], dim=1)
            elif 'Block' in module.__class__.__name__:
                x = (x,) + input[1:]
            x = module(x)
        return x


# Define a resnet block
class ResnetBlock(nn.Module):
    def __init__(self, dim, norm_layer, use_dropout, use_bias, padding_type='reflect', n_domains=0):
        super(ResnetBlock, self).__init__()

        conv_block = []
        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)

        conv_block += [nn.Conv2d(dim + n_domains, dim, kernel_size=3, padding=p, bias=use_bias),
                       norm_layer(dim),
                       nn.PReLU()]
        if use_dropout:
            conv_block += [nn.Dropout(0.5)]

        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)
        conv_block += [nn.Conv2d(dim + n_domains, dim, kernel_size=3, padding=p, bias=use_bias),
                       norm_layer(dim)]

        self.conv_block = SequentialContext(n_domains, *conv_block)

    def forward(self, input):
        if isinstance(input, tuple):
            return input[0] + self.conv_block(*input)
        return input + self.conv_block(input)


class ResnetGenEncoderv1(nn.Module):
    def __init__(self, input_nc, n_blocks=4, ngf=64, norm_layer=nn.BatchNorm2d,
                 use_dropout=False, gpu_ids=[], use_bias=False, padding_type='reflect'):
        assert(n_blocks >= 0)
        super(ResnetGenEncoderv1, self).__init__()
        self.gpu_ids = gpu_ids

        model = [nn.ReflectionPad2d(3),
                 nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0,
                           bias=use_bias),
                 norm_layer(ngf),
                 nn.PReLU()]

        n_downsampling = 2
        for i in range(n_downsampling):
            mult = 2**i
            model += [nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3,
                                stride=2, padding=1, bias=use_bias),
                      norm_layer(ngf * mult * 2),
                      nn.PReLU()]

        mult = 2**n_downsampling
        model += [PGAResBlockv4k3(ngf * mult, norm_layer=norm_layer, use_bias=use_bias)]
        model_res = []
        for _ in range(n_blocks):
            model_res += [ResnetBlock(ngf * mult, norm_layer=norm_layer,
                                  use_dropout=use_dropout, use_bias=use_bias, padding_type=padding_type)]

        self.model = nn.Sequential(*model)
        self.model_postfix = nn.Sequential(*model_res)

    def forward(self, input):
        # if self.gpu_ids and isinstance(input.data, torch.cuda.FloatTensor):
        #     out1, attmap1, attmap2, attmap3, attmap4 = nn.parallel.data_parallel(self.model, input, self.gpu_ids)
        #     out2 = nn.parallel.data_parallel(self.model_postfix, out1, self.gpu_ids)
        # else:
        out1, attmap1, attmap2, attmap3, attmap4 = self.model(input)
        out2 = self.model_postfix(out1)
        return out2, attmap1, attmap2, attmap3, attmap4


class ResnetGenDecoderv1(nn.Module):
    def __init__(self, output_nc, n_blocks=5, ngf=64, norm_layer=nn.BatchNorm2d,
                 use_dropout=False, gpu_ids=[], use_bias=False, padding_type='reflect'):
        assert(n_blocks >= 0)
        super(ResnetGenDecoderv1, self).__init__()
        self.gpu_ids = gpu_ids
        
        model = []
        n_downsampling = 2
        mult = 2**n_downsampling
        self.model_att = PGAResBlockv4k3(ngf * mult, norm_layer=norm_layer, use_bias=use_bias)

        for _ in range(n_blocks):
            model += [ResnetBlock(ngf * mult, norm_layer=norm_layer,
                                  use_dropout=use_dropout, use_bias=use_bias, padding_type=padding_type)]
            
        for i in range(n_downsampling):
            mult = 2**(n_downsampling - i)
            model += [nn.ConvTranspose2d(ngf * mult, int(ngf * mult / 2),
                                         kernel_size=4, stride=2,
                                         padding=1, output_padding=0,
                                         bias=use_bias),
                      nn.GroupNorm(32, int(ngf * mult / 2)),
                      nn.PReLU()]

        model += [nn.ReflectionPad2d(3),
                  nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0),
                  nn.Tanh()]

        self.model = nn.Sequential(*model)

    def forward(self, input):
        if self.gpu_ids and isinstance(input.data, torch.cuda.FloatTensor):
            out1, _, _, _, _ = nn.parallel.data_parallel(self.model_att, input, self.gpu_ids)
            out2 = nn.parallel.data_parallel(self.model, out1, self.gpu_ids)
        else:
            out1, _, _, _, _ = self.model_att(input)
            out2 = self.model(out1)

        return out2


class PEARLGAN(nn.Module):
    def __init__(self, input_nc=3, output_nc=3, ngf=64, n_blocks=9, n_blocks_shared=0, norm='instance', use_dropout=False, gpu_ids=[], ckpt_path='./pearlgan/checkpoints/FLIR_NTIR2DC'):
        super().__init__()

        norm_layer = get_norm_layer(norm_type=norm)
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        n_blocks -= n_blocks_shared
        n_blocks_enc = n_blocks // 2
        n_blocks_dec = n_blocks - n_blocks_enc

        dup_args = (ngf, norm_layer, use_dropout, gpu_ids, use_bias)
        enc_args = (input_nc, n_blocks_enc) + dup_args
        dec_args = (output_nc, n_blocks_dec) + dup_args

        self.DA = 1  # useless but keep it
        self.d = 0   # useless but keep it


        self.encoder = ResnetGenEncoderv1(*enc_args)
        self.decoder = ResnetGenDecoderv1(*dec_args)
        if os.path.exists(os.path.join(ckpt_path, '80_net_G1.pth')):
            self.encoder.load_state_dict(torch.load(os.path.join(ckpt_path, '80_net_G1.pth')))
            self.decoder.load_state_dict(torch.load(os.path.join(ckpt_path, '80_net_G2.pth')))
        elif os.path.exists(os.path.join(ckpt_path, '120_net_G1.pth')):
            self.encoder.load_state_dict(torch.load(os.path.join(ckpt_path, '120_net_G1.pth')))
            self.decoder.load_state_dict(torch.load(os.path.join(ckpt_path, '120_net_G2.pth')))
        else:
            raise ValueError(f"No valid pearlgan ckpts under {ckpt_path}.")
        
    def forward(self, input):
        encoded, _, _, _, _ = self.encoder(input)
        translated_vis = self.decoder(encoded)

        return encoded, translated_vis


################################################### PEARLGAN ###################################################




class Focus(nn.Module):
    """Focus width and height information into channel space.

    Args:
        in_channels (int): The input channels of this Module.
        out_channels (int): The output channels of this Module.
        kernel_size (int): The kernel size of the convolution. Default: 1
        stride (int): The stride of the convolution. Default: 1
        conv_cfg (dict): Config dict for convolution layer. Default: None,
            which means using conv2d.
        norm_cfg (dict): Config dict for normalization layer.
            Default: dict(type='BN', momentum=0.03, eps=0.001).
        act_cfg (dict): Config dict for activation layer.
            Default: dict(type='Swish').
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size=1,
                 stride=1,
                 conv_cfg=None,
                 norm_cfg=dict(type='BN', momentum=0.03, eps=0.001),
                 act_cfg=dict(type='Swish')):
        super().__init__()
        self.conv = ConvModule(
            in_channels * 4,
            out_channels,
            kernel_size,
            stride,
            padding=(kernel_size - 1) // 2,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg)

    def forward(self, x):
        # shape of x (b,c,w,h) -> y(b,4c,w/2,h/2)
        patch_top_left = x[..., ::2, ::2]
        patch_top_right = x[..., ::2, 1::2]
        patch_bot_left = x[..., 1::2, ::2]
        patch_bot_right = x[..., 1::2, 1::2]
        x = torch.cat(
            (
                patch_top_left,
                patch_bot_left,
                patch_top_right,
                patch_bot_right,
            ),
            dim=1,
        )
        return self.conv(x)


class SPPBottleneck(BaseModule):
    """Spatial pyramid pooling layer used in YOLOv3-SPP.

    Args:
        in_channels (int): The input channels of this Module.
        out_channels (int): The output channels of this Module.
        kernel_sizes (tuple[int]): Sequential of kernel sizes of pooling
            layers. Default: (5, 9, 13).
        conv_cfg (dict): Config dict for convolution layer. Default: None,
            which means using conv2d.
        norm_cfg (dict): Config dict for normalization layer.
            Default: dict(type='BN').
        act_cfg (dict): Config dict for activation layer.
            Default: dict(type='Swish').
        init_cfg (dict or list[dict], optional): Initialization config dict.
            Default: None.
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_sizes=(5, 9, 13),
                 conv_cfg=None,
                 norm_cfg=dict(type='BN', momentum=0.03, eps=0.001),
                 act_cfg=dict(type='Swish'),
                 init_cfg=None):
        super().__init__(init_cfg)
        mid_channels = in_channels // 2
        self.conv1 = ConvModule(
            in_channels,
            mid_channels,
            1,
            stride=1,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg)
        self.poolings = nn.ModuleList([
            nn.MaxPool2d(kernel_size=ks, stride=1, padding=ks // 2)
            for ks in kernel_sizes
        ])
        conv2_channels = mid_channels * (len(kernel_sizes) + 1)
        self.conv2 = ConvModule(
            conv2_channels,
            out_channels,
            1,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg)

    def forward(self, x):
        x = self.conv1(x)
        x = torch.cat([x] + [pooling(x) for pooling in self.poolings], dim=1)
        x = self.conv2(x)
        return x


@BACKBONES.register_module()
class CSPDarknet_TIR(BaseModule):
    """CSP-Darknet backbone used in YOLOv5 and YOLOX.

    Args:
        arch (str): Architecture of CSP-Darknet, from {P5, P6}.
            Default: P5.
        deepen_factor (float): Depth multiplier, multiply number of
            blocks in CSP layer by this amount. Default: 1.0.
        widen_factor (float): Width multiplier, multiply number of
            channels in each layer by this amount. Default: 1.0.
        out_indices (Sequence[int]): Output from which stages.
            Default: (2, 3, 4).
        frozen_stages (int): Stages to be frozen (stop grad and set eval
            mode). -1 means not freezing any parameters. Default: -1.
        use_depthwise (bool): Whether to use depthwise separable convolution.
            Default: False.
        arch_ovewrite(list): Overwrite default arch settings. Default: None.
        spp_kernal_sizes: (tuple[int]): Sequential of kernel sizes of SPP
            layers. Default: (5, 9, 13).
        conv_cfg (dict): Config dict for convolution layer. Default: None.
        norm_cfg (dict): Dictionary to construct and config norm layer.
            Default: dict(type='BN', requires_grad=True).
        act_cfg (dict): Config dict for activation layer.
            Default: dict(type='LeakyReLU', negative_slope=0.1).
        norm_eval (bool): Whether to set norm layers to eval mode, namely,
            freeze running stats (mean and var). Note: Effect on Batch Norm
            and its variants only.
        init_cfg (dict or list[dict], optional): Initialization config dict.
            Default: None.
    Example:
        >>> from mmdet.models import CSPDarknet
        >>> import torch
        >>> self = CSPDarknet(depth=53)
        >>> self.eval()
        >>> inputs = torch.rand(1, 3, 416, 416)
        >>> level_outputs = self.forward(inputs)
        >>> for level_out in level_outputs:
        ...     print(tuple(level_out.shape))
        ...
        (1, 256, 52, 52)
        (1, 512, 26, 26)
        (1, 1024, 13, 13)
    """
    # From left to right:
    # in_channels, out_channels, num_blocks, add_identity, use_spp
    arch_settings = {
        'P5': [[64, 128, 3, True, False], [128, 256, 9, True, False],
               [256, 512, 9, True, False], [512, 1024, 3, False, True]],
        'P6': [[64, 128, 3, True, False], [128, 256, 9, True, False],
               [256, 512, 9, True, False], [512, 768, 3, True, False],
               [768, 1024, 3, False, True]]
    }

    def __init__(self,
                 arch='P5',
                 deepen_factor=1.0,
                 widen_factor=1.0,
                 out_indices=(2, 3, 4),
                 frozen_stages=-1,
                 use_depthwise=False,
                 arch_ovewrite=None,
                 spp_kernal_sizes=(5, 9, 13),
                 conv_cfg=None,
                 norm_cfg=dict(type='BN', momentum=0.03, eps=0.001),
                 act_cfg=dict(type='Swish'),
                 norm_eval=False,
                 pearlgan_ckpt='configs/tirdet/pearlgan_ckpt/FLIR_NTIR2DC/',
                 pearlgan_half=True,
                 freeze_pearlgan=True,
                 init_cfg=dict(
                     type='Kaiming',
                     layer='Conv2d',
                     a=math.sqrt(5),
                     distribution='uniform',
                     mode='fan_in',
                     nonlinearity='leaky_relu')):
        super().__init__(init_cfg)
        arch_setting = self.arch_settings[arch]
        if arch_ovewrite:
            arch_setting = arch_ovewrite
        assert set(out_indices).issubset(
            i for i in range(len(arch_setting) + 1))
        if frozen_stages not in range(-1, len(arch_setting) + 1):
            raise ValueError('frozen_stages must be in range(-1, '
                             'len(arch_setting) + 1). But received '
                             f'{frozen_stages}')

        self.out_indices = out_indices
        self.frozen_stages = frozen_stages
        self.use_depthwise = use_depthwise
        self.norm_eval = norm_eval
        self.pearlgan_half = pearlgan_half
        conv = DepthwiseSeparableConvModule if use_depthwise else ConvModule

        self.pearlgan = PEARLGAN(ckpt_path=pearlgan_ckpt) ### added
        if pearlgan_half:
            self.pearlgan = self.pearlgan.half()
        if freeze_pearlgan:
            for p in self.pearlgan.parameters():
                p.requires_grad = False

        self.stem = Focus(
            in_channels=6,
            out_channels=int(arch_setting[0][0] * widen_factor),
            kernel_size=3,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg)
        self.layers = ['stem']

        for i, (in_channels, out_channels, num_blocks, add_identity,
                use_spp) in enumerate(arch_setting):
            in_channels = int(in_channels * widen_factor)
            out_channels = int(out_channels * widen_factor)
            num_blocks = max(round(num_blocks * deepen_factor), 1)
            stage = []
            conv_layer = conv(
                in_channels,
                out_channels,
                3,
                stride=2,
                padding=1,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg,
                act_cfg=act_cfg)
            stage.append(conv_layer)
            if use_spp:
                spp = SPPBottleneck(
                    out_channels,
                    out_channels,
                    kernel_sizes=spp_kernal_sizes,
                    conv_cfg=conv_cfg,
                    norm_cfg=norm_cfg,
                    act_cfg=act_cfg)
                stage.append(spp)
            csp_layer = CSPLayer(
                out_channels,
                out_channels,
                num_blocks=num_blocks,
                add_identity=add_identity,
                use_depthwise=use_depthwise,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg,
                act_cfg=act_cfg)
            stage.append(csp_layer)
            self.add_module(f'stage{i + 1}', nn.Sequential(*stage))
            self.layers.append(f'stage{i + 1}')

    def _freeze_stages(self):
        if self.frozen_stages >= 0:
            for i in range(self.frozen_stages + 1):
                m = getattr(self, self.layers[i])
                m.eval()
                for param in m.parameters():
                    param.requires_grad = False

    def train(self, mode=True):
        super(CSPDarknet_TIR, self).train(mode)
        self._freeze_stages()
        if mode and self.norm_eval:
            for m in self.modules():
                if isinstance(m, _BatchNorm):
                    m.eval()

    def forward(self, x):
        outs = []
        if self.pearlgan_half:
            z, translated_vis = self.pearlgan(x.half())
            z, translated_vis = z.float(), translated_vis.float()
        else:
            z, translated_vis = self.pearlgan(x) 
        x = torch.cat((x, translated_vis), dim=1) 
        for i, layer_name in enumerate(self.layers):
            layer = getattr(self, layer_name)
            x = layer(x)
            if i in self.out_indices:
                outs.append(x)
        return (z, outs)
