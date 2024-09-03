#! /usr/bin/env python
# -*- coding: utf-8 -*-

# Copyright 2021 Imperial College London (Pingchuan Ma)
# Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

import torch
import torch.nn as nn
from espnet.nets.pytorch_backend.backbones.modules.resnet_udp import BasicBlock_udp, ResNet_udp
from espnet.nets.pytorch_backend.transformer.convolution import Swish
from espnet.nets.pytorch_backend.backbones.conv3d_extractor import threeD_to_2D_tensor



class Conv3dResNet_udp(torch.nn.Module):
    """Conv3dResNet user adaptation padding module"""

    def __init__(self, backbone_type="resnet", relu_type="swish"):
        """__init__.

        :param backbone_type: str, the type of a visual front-end.
        :param relu_type: str, activation function used in an audio front-end.
        """
        super(Conv3dResNet_udp, self).__init__()
        self.frontend_nout = 64
        self.trunk = ResNet_udp(BasicBlock_udp, [2, 2, 2, 2], relu_type=relu_type)
        self.frontend3D = nn.Sequential(
            nn.Conv3d(
                1, self.frontend_nout, (5, 7, 7), (1, 2, 2), padding=(0, 0, 0), bias=False
            ),
            nn.BatchNorm3d(self.frontend_nout),
            Swish(),
            nn.MaxPool3d((1, 3, 3), (1, 2, 2), (0, 1, 1)),
        )

    def forward(self, xs_pad, udp):
        xs_pad = xs_pad.transpose(1, 2)  # [B, T, C, H, W] -> [B, C, T, H, W]
        xs_pad = self.attach_udp_input_3d(2, 3, xs_pad, udp['udp0'].cuda())

        B, C, T, H, W = xs_pad.size()
        xs_pad = self.frontend3D(xs_pad)
        Tnew = xs_pad.shape[2]
        xs_pad = threeD_to_2D_tensor(xs_pad)
        xs_pad = self.trunk(xs_pad,udp)
        return xs_pad.view(B, Tnew, xs_pad.size(1))
    
    def attach_udp_input_3d(self, pad_t, pad_s, x, udp):
        # x: B, C, T, H, W
        feat_size_t, feat_size_h, feat_size_w = [*x.size()[-3:]]
        meta_frame = torch.zeros([x.size(1), feat_size_t + pad_t * 2, feat_size_h + pad_s * 2, feat_size_w + pad_s * 2],dtype=udp.dtype).cuda()
        index = torch.ones_like(meta_frame)
        index[:, :, pad_s:feat_size_h + pad_s, pad_s:feat_size_w + pad_s] = 0.
        index = index.int().bool()
        meta_frame[index] = udp.repeat(x.size(2) + pad_t * 2)
        meta_framed = meta_frame.unsqueeze(0).repeat(x.size(0), 1, 1, 1, 1)
        meta_framed[:, :, pad_t:feat_size_t + pad_t, pad_s:feat_size_h + pad_s, pad_s:feat_size_w + pad_s] = x
        meta_framed[:, :, :pad_t, :, :] = 0.
        meta_framed[:, :, feat_size_t + pad_t:, :, :] = 0.
        return meta_framed
