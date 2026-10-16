#!/usr/bin/python3.10.1 (python版本)
# -*- coding: utf-8 -*-
# @Author  : Jiangxiong Fang
# @Email   : fangchj2002@163.com

# @Time    : 2023/12/10 上午9:39
from nnunetv2.training.nnUNetTrainer.others.CNN_UNet.our_FourierNet.fft.SFBlock import *
import torch.nn.functional as F
import torch.nn as nn
import torch

class DoubleConv_block(nn.Module):
    """
    Convolution Block
    """

    def __init__(self, in_ch, out_ch):
        super(DoubleConv_block, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True))

    def forward(self, x):
        x = self.conv(x)
        return x

class LayerNorm(nn.Module):
    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError
        self.normalized_shape = (normalized_shape,)

    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x
class up_conv(nn.Module):
    """
    Up Convolution Block
    """
    def __init__(self, in_ch, out_ch):
        super(up_conv, self).__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2),

            nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.up(x)
        return x
class group_aggregation_bridge(nn.Module):
    def __init__(self, dim_xh, dim_xl,dim_mixed, k_size=3, d_list=[1, 2, 5, 7]):
        super().__init__()
        self.pre_project = nn.Conv2d(dim_xh, dim_xl, 1)
        group_size = dim_xl // 2
        dim_mk = dim_mixed // 4
        self.g0 = nn.Sequential(
            LayerNorm(normalized_shape=group_size + dim_mk, data_format='channels_first'),
            nn.Conv2d(group_size + dim_mk, group_size + dim_mk, kernel_size=3, stride=1,
                      padding=(k_size+(k_size-1)*(d_list[0]-1))//2,
                      dilation=d_list[0], groups=group_size + dim_mk)
        )
        self.g1 = nn.Sequential(
            LayerNorm(normalized_shape=group_size+dim_mk, data_format='channels_first'),
            nn.Conv2d(group_size + dim_mk, group_size + dim_mk, kernel_size=3, stride=1,
                      padding=(k_size+(k_size-1)*(d_list[1]-1))//2,
                      dilation=d_list[1], groups=group_size + dim_mk)
        )
        self.g2 = nn.Sequential(
            LayerNorm(normalized_shape=group_size + dim_mk, data_format='channels_first'),
            nn.Conv2d(group_size + dim_mk, group_size + dim_mk, kernel_size=3, stride=1,
                      padding=(k_size+(k_size-1)*(d_list[2]-1))//2,
                      dilation=d_list[2], groups=group_size + dim_mk)
        )
        self.g3 = nn.Sequential(
            LayerNorm(normalized_shape=group_size+dim_mk, data_format='channels_first'),
            nn.Conv2d(group_size + dim_mk, group_size + dim_mk, kernel_size=3, stride=1,
                      padding=(k_size+(k_size-1)*(d_list[3]-1))//2,
                      dilation=d_list[3], groups=group_size + dim_mk)
        )
        self.tail_conv = nn.Sequential(
            LayerNorm(normalized_shape=dim_xl * 2 + dim_mixed, data_format='channels_first'),
            nn.Conv2d(dim_xl * 2 + dim_mixed, dim_xl, 1)
        )
    def forward(self, xh, xl, mk):
        xh = self.pre_project(xh)
        xh = F.interpolate(xh, size=[xl.size(2), xl.size(3)], mode ='bilinear', align_corners=True)
        xh = torch.chunk(xh, 4, dim=1)
        xl = torch.chunk(xl, 4, dim=1)
        mk = torch.chunk(mk, 4, dim=1)
        x0 = self.g0(torch.cat((xh[0], xl[0], mk[0]), dim=1))
        x1 = self.g1(torch.cat((xh[1], xl[1], mk[1]), dim=1))
        x2 = self.g2(torch.cat((xh[2], xl[2], mk[2]), dim=1))
        x3 = self.g3(torch.cat((xh[3], xl[3], mk[3]), dim=1))
        x = torch.cat((x0,x1,x2,x3), dim=1)
        x = self.tail_conv(x)
        return x

class Grouped_multi_axis_Hadamard_Product_Attention(nn.Module):
    def __init__(self, dim_in, dim_out, x=8, y=8):
        super().__init__()

        c_dim_in = dim_in // 4
        k_size = 3
        pad = (k_size - 1) // 2

        self.params_xy = nn.Parameter(torch.Tensor(1, c_dim_in, x, y), requires_grad=True)
        nn.init.ones_(self.params_xy)
        self.conv_xy = nn.Sequential(nn.Conv2d(c_dim_in, c_dim_in, kernel_size=k_size, padding=pad, groups=c_dim_in),
                                     nn.GELU(), nn.Conv2d(c_dim_in, c_dim_in, 1))

        self.params_zx = nn.Parameter(torch.Tensor(1, 1, c_dim_in, x), requires_grad=True)
        nn.init.ones_(self.params_zx)
        self.conv_zx = nn.Sequential(nn.Conv1d(c_dim_in, c_dim_in, kernel_size=k_size, padding=pad, groups=c_dim_in),
                                     nn.GELU(), nn.Conv1d(c_dim_in, c_dim_in, 1))

        self.params_zy = nn.Parameter(torch.Tensor(1, 1, c_dim_in, y), requires_grad=True)
        nn.init.ones_(self.params_zy)
        self.conv_zy = nn.Sequential(nn.Conv1d(c_dim_in, c_dim_in, kernel_size=k_size, padding=pad, groups=c_dim_in),
                                     nn.GELU(), nn.Conv1d(c_dim_in, c_dim_in, 1))

        self.dw = nn.Sequential(
            nn.Conv2d(c_dim_in, c_dim_in, 1),
            nn.GELU(),
            nn.Conv2d(c_dim_in, c_dim_in, kernel_size=3, padding=1, groups=c_dim_in)
        )

        self.norm1 = LayerNorm(dim_in, eps=1e-6, data_format='channels_first')
        self.norm2 = LayerNorm(dim_in, eps=1e-6, data_format='channels_first')

        self.ldw = nn.Sequential(
            nn.Conv2d(dim_in, dim_in, kernel_size=3, padding=1, groups=dim_in),
            nn.GELU(),
            nn.Conv2d(dim_in, dim_out, 1),
        )

    def forward(self, x):
        x = self.norm1(x)
        x1, x2, x3, x4 = torch.chunk(x, 4, dim=1)
        # B, C, H, W = x1.size()
        params_xy = self.params_xy
        params_zx = self.params_zx
        params_zy = self.params_zy
        # ----------x1----------#
        # ----------xy----------#
        x1_1 = x1 * self.conv_xy(F.interpolate(params_xy, size=x1.shape[2:4], mode='bilinear', align_corners=True))
        # ----------zx----------#
        x1_a = x1.permute(0, 3, 1, 2)
        x1_2 = x1_a * self.conv_zx(F.interpolate(params_zx, size=x1_a.shape[2:4], mode='bilinear', align_corners=True).squeeze(0)).unsqueeze(0)
        x1_2 = x1_2.permute(0, 2, 3, 1)
        # ----------zy----------#
        x1_b = x1.permute(0, 2, 1, 3)
        x1_3 = x1_b * self.conv_zy(F.interpolate(params_zy, size=x1_b.shape[2:4], mode='bilinear', align_corners=True).squeeze(0)).unsqueeze(0)
        x1_3 = x1_3.permute(0, 2, 1, 3)

        x1 = x1_1 + x1_2 + x1_3 + x1
        # ----------x2----------#
        # ----------xy----------#
        x2_1 = x2 * self.conv_xy(F.interpolate(params_xy, size=x2.shape[2:4], mode='bilinear', align_corners=True))
        # ----------zx----------#
        x2_a = x2.permute(0, 3, 1, 2)
        x2_2 = x2_a * self.conv_zx(F.interpolate(params_zx, size=x2_a.shape[2:4], mode='bilinear', align_corners=True).squeeze(0)).unsqueeze(0)
        x2_2 = x2_2.permute(0, 2, 3, 1)
        # ----------zy----------#
        x2_b = x2.permute(0, 2, 1, 3)
        x2_3 = x2_b * self.conv_zy(F.interpolate(params_zy, size=x2_b.shape[2:4], mode='bilinear', align_corners=True).squeeze(0)).unsqueeze(0)
        x2_3 = x2_3.permute(0, 2, 1, 3)

        x2 = x2_1 + x2_2 + x2_3 + x2
        # ----------x3----------#
        # ----------xy----------#
        x3_1 = x3 * self.conv_xy(F.interpolate(params_xy, size=x3.shape[2:4], mode='bilinear', align_corners=True))
        # ----------zx----------#
        x3_a = x3.permute(0, 3, 1, 2)
        x3_2 = x3_a * self.conv_zx(F.interpolate(params_zx, size=x1_a.shape[2:4], mode='bilinear', align_corners=True).squeeze(0)).unsqueeze(0)
        x3_2 = x3_2.permute(0, 2, 3, 1)
        # ----------zy----------#
        x3_b = x3.permute(0, 2, 1, 3)
        x3_3 = x3_b * self.conv_zy(F.interpolate(params_zy, size=x1_b.shape[2:4], mode='bilinear', align_corners=True).squeeze(0)).unsqueeze(0)
        x3_3 = x3_3.permute(0, 2, 1, 3)

        x3 = x3_1 + x3_2 + x3_3 + x3
        x4 = self.dw(x4) + x4
        # ----------concat----------#
        x = torch.cat([x1, x2, x3, x4], dim=1)
        # ----------ldw----------#
        x = self.norm2(x)
        x = self.ldw(x)
        return x

class SpectralUNet_DouleBranch_layer5(nn.Module):
    def __init__(self, in_ch=1, out_ch=3, nf=32):
        super(SpectralUNet_DouleBranch_layer5, self).__init__()

        self.nf = nf * 2

        filters = [nf, nf * 2, nf * 4, nf * 8, nf * 16]
        self.Fre_conv_first = nn.Conv2d(in_ch, 3, 3, 1, 1, bias=True)
        self.inconv0 = DoubleConv_block(in_ch, filters[0])
        self.Fre_inconv0 = DoubleConv_block(2 * 2, filters[0])
        self.inconv1 = DoubleConv_block(filters[0], filters[1])
        self.inconv2 = DoubleConv_block(filters[1], filters[2])
        self.inconv3 = DoubleConv_block(filters[2], filters[3])
        self.inconv4 = DoubleConv_block(filters[3], filters[4])

        self.Maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Maxpool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Maxpool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.apectral_spatial_layer5 = ProcessBlock(filters[4], spatial=False)
        self.apectral_spatial_layer4 = ProcessBlock(filters[3], spatial=False)
        self.apectral_spatial_layer3 = ProcessBlock(filters[2], spatial=False)
        self.apectral_spatial_layer2 = ProcessBlock(filters[1], spatial=False)

        self.Conv = nn.Conv2d(filters[0], out_ch, kernel_size=1, stride=1, padding=0)

        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)

        self.ebn1 = nn.GroupNorm(4, filters[1])
        self.ebn2 = nn.GroupNorm(4, filters[2])
        self.ebn3 = nn.GroupNorm(4, filters[3])
        self.ebn4 = nn.GroupNorm(4, filters[4])

        self.encoder1 = nn.Sequential(
            Grouped_multi_axis_Hadamard_Product_Attention(filters[0], filters[1]),
        )
        self.encoder2 = nn.Sequential(
            Grouped_multi_axis_Hadamard_Product_Attention(filters[1], filters[2]),
        )
        self.encoder3 = nn.Sequential(
            Grouped_multi_axis_Hadamard_Product_Attention(filters[2], filters[3]),
        )
        self.encoder4 = nn.Sequential(
            Grouped_multi_axis_Hadamard_Product_Attention(filters[3], filters[4]),
        )
        self.layer5_gt = nn.Conv2d(filters[4], filters[4], 3, stride=1, padding=1)
        self.layer4_gt = nn.Conv2d(filters[4], filters[4], 3, stride=1, padding=1)
        self.layer3_gt = nn.Conv2d(filters[3], filters[3], 3, stride=1, padding=1)
        self.layer2_gt = nn.Conv2d(filters[2], filters[2], 3, stride=1, padding=1)
        self.layer1_gt = nn.Conv2d(filters[1], filters[1], 3, stride=1, padding=1)
        self.GAB5 = group_aggregation_bridge(filters[4], filters[4], filters[4])
        self.GAB4 = group_aggregation_bridge(filters[4], filters[3], filters[3])
        self.GAB3 = group_aggregation_bridge(filters[3], filters[2], filters[2])
        self.GAB2 = group_aggregation_bridge(filters[2], filters[1], filters[1])
        self.GAB1 = group_aggregation_bridge(filters[1], filters[0], filters[0])

    def forward(self, x):
        _, _, H, W = x.shape
        image_fft = torch.fft.fft2(x, norm='backward')
        mag_image = torch.abs(image_fft)
        pha_image = torch.angle(image_fft)
        real_image_enhanced = mag_image * torch.cos(pha_image)
        imag_image_enhanced = mag_image * torch.sin(pha_image)
        img_amp_enhanced = torch.fft.ifft2(torch.complex(real_image_enhanced, imag_image_enhanced), s=(H, W),
                                           norm='backward').real

        x_middle = img_amp_enhanced

        rate = 2 ** 3
        pad_h = (rate - H % rate) % rate
        pad_w = (rate - W % rate) % rate

        if pad_h != 0 or pad_w != 0:
            x_middle = F.pad(x_middle , (0, pad_w, 0, pad_h), "reflect")
            x = F.pad(x, (0, pad_w, 0, pad_h), "reflect")
        x_fea = self.Fre_conv_first(x)
        L1_fea = self.lrelu(torch.cat((x_middle, x_fea), dim=1))

        Fre_e1 = self.Fre_inconv0(L1_fea)

        Fre_e2 = self.Maxpool1(Fre_e1)
        Fre_e2 = self.inconv1(Fre_e2)

        # Fre_e2 = self.apectral_spatial_layer2(Fre_e2)
        Fre_e3 = self.Maxpool2(Fre_e2)
        Fre_e3 = self.inconv2(Fre_e3)

        Fre_e3 = self.apectral_spatial_layer3(Fre_e3)
        Fre_e4 = self.Maxpool3(Fre_e3)
        Fre_e4 = self.inconv3(Fre_e4)

        # Fre_e4 = self.apectral_spatial_layer4(Fre_e4)
        Fre_e5 = self.Maxpool4(Fre_e4)

        Fre_e5 = self.inconv4(Fre_e5)

        Fre_e5 = self.apectral_spatial_layer5(Fre_e5)

        e1 = self.inconv0(x)

        # e2 = F.max_pool2d(self.ebn1(self.encoder1(e1)), 2, 2)
        e2 = self.Maxpool1(e1)
        e2 = self.inconv1(e2)
        #
        e3 = F.max_pool2d(self.ebn2(self.encoder2(e2)), 2, 2)
        # e3 = self.Maxpool2(e2)
        # e3 = self.inconv2(e3)
        #
        # e4 = F.max_pool2d(self.ebn3(self.encoder3(e3)), 2, 2)
        e4 = self.Maxpool3(e3)
        e4 = self.inconv3(e4)

        e5 = F.max_pool2d(self.ebn4(self.encoder4(e4)), 2, 2)

        gt_layer5 = self.layer5_gt(e5 + Fre_e5)
        t5 = self.GAB5(gt_layer5, e5, Fre_e5)

        d4 = F.interpolate(t5, scale_factor=2, mode='bilinear', align_corners=True)
        gt_layer4 = self.layer4_gt(d4)
        t4 = self.GAB4(gt_layer4, e4, Fre_e4)

        d3 = F.interpolate(t4, scale_factor=2, mode='bilinear', align_corners=True)
        gt_layer3 = self.layer3_gt(d3)
        t3 = self.GAB3(gt_layer3, e3, Fre_e3)

        d2 = F.interpolate(t3, scale_factor=2, mode='bilinear', align_corners=True)
        gt_layer2 = self.layer2_gt(d2)
        t2 = self.GAB2(gt_layer2, e2, Fre_e2)

        d1 = F.interpolate(t2, scale_factor=2, mode='bilinear', align_corners=True)
        gt_layer1 = self.layer1_gt(d1)
        t1 = self.GAB1(gt_layer1, e1, Fre_e1)

        out = self.Conv(t1)

        return out

# network = SpectralUNet_DouleBranch_layer5(1, 2)
# x = torch.rand(2, 1, 512, 512)
# network(x)
# total = sum([param.nelement() for param in network.parameters()])
# print("Number of parameter: %.2fM" % (total/1e6))


# -- coding: utf-8 --
# import torch
# from thop import profile
#
# # Model
# print('==> Building model..')
# network = SpectralUNet_DouleBranch_layer5(1, 3)
# dummy_input = torch.rand(2, 1, 512, 512)
# flops, params = profile(network, (dummy_input,))
# print('flops: ', flops, 'params: ', params)
# print('flops: %.2f M, params: %.2f M' % (flops / 1000000.0, params / 1000000.0))
