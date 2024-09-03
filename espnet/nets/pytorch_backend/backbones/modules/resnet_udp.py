import math
import pdb
import torch

import torch.nn as nn

from espnet.nets.pytorch_backend.transformer.convolution import Swish
from espnet.nets.pytorch_backend.backbones.modules.resnet import downsample_basic_block

def conv3x3_udp(in_planes, out_planes, stride=1):
    """conv3x3.

    :param in_planes: int, number of channels in the input sequence.
    :param out_planes: int,  number of channels produced by the convolution.
    :param stride: int, size of the convolving kernel.
    """
    return nn.Conv2d(
        in_planes,
        out_planes,
        kernel_size=3,
        stride=stride,
        padding=0,
        bias=False,
    )


class BasicBlock_udp(nn.Module):
    expansion = 1

    def __init__(
        self,
        inplanes,
        planes,
        stride=1,
        downsample=None,
        relu_type="swish",
    ):
        """__init__.

        :param inplanes: int, number of channels in the input sequence.
        :param planes: int,  number of channels produced by the convolution.
        :param stride: int, size of the convolving kernel.
        :param downsample: boolean, if True, the temporal resolution is downsampled.
        :param relu_type: str, type of activation function.
        """
        super(BasicBlock_udp, self).__init__()

        assert relu_type in ["relu", "prelu", "swish"]
        self.conv1 = conv3x3_udp(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = conv3x3_udp(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        
        
        if relu_type == "relu":
            self.relu1 = nn.ReLU(inplace=True)
            self.relu2 = nn.ReLU(inplace=True)
        elif relu_type == "prelu":
            self.relu1 = nn.PReLU(num_parameters=planes)
            self.relu2 = nn.PReLU(num_parameters=planes)
        elif relu_type == "swish":
            self.relu1 = Swish()
            self.relu2 = Swish()
        else:
            raise NotImplementedError
        
        self.downsample = downsample
        self.stride = stride

    def forward(self, x, udp1, udp2):
        """forward.

        :param x: torch.Tensor, input tensor with input size (B, C, T, H, W).
        """

        residual = x
        out = self.conv1(attach_udp_input(x, udp1))
        out = self.bn1(out)
        out = self.relu1(out)
        out = self.conv2(attach_udp_input(out, udp2))
        out = self.bn2(out)
        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu2(out)

        return out

class mySequential_bottleneck(nn.Sequential):
    def forward(self, *input):
        x = input[0]
        for i, module in enumerate(self._modules.values()):
            if i == 0:
                x = attach_udp_input(x, input[1])
                x = module(x)
            elif i == 3:
                x = attach_udp_input(x, input[2])
                x = module(x)
            else:
                x = module(x)
        return x


def attach_udp_input(x, udp, pad=1):
    feat_size = x.size(2)
    meta_frame = torch.zeros([x.size(1), feat_size + pad * 2, feat_size + pad * 2],dtype=udp.dtype).cuda()
    index = torch.ones_like(meta_frame)
    index[:, pad:feat_size + pad, pad:feat_size + pad] = 0
    index = index.int().bool()
    meta_frame[index] = udp
    meta_framed = meta_frame.unsqueeze(0).repeat(x.size(0), 1, 1, 1)
    meta_framed[:, :, pad:feat_size + pad, pad:feat_size + pad] = x
    return meta_framed

class ResNet_udp(nn.Module):
    def __init__(
        self,
        block,
        layers,
        relu_type="swish",
    ):
        super(ResNet_udp, self).__init__()
        self.inplanes = 64
        self.relu_type = relu_type
        self.downsample_block = downsample_basic_block

        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d(1)

    def _make_layer(self, block, planes, blocks, stride=1):
        """_make_layer.

        :param block: torch.nn.Module, class of blocks.
        :param planes: int,  number of channels produced by the convolution.
        :param blocks: int, number of layers in a block.
        :param stride: int, size of the convolving kernel.
        """
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = self.downsample_block(
                inplanes=self.inplanes,
                outplanes=planes * block.expansion,
                stride=stride,
            )

        layers = []
        layers.append(
            block(
                self.inplanes,
                planes,
                stride,
                downsample,
                relu_type=self.relu_type,
            )
        )
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(
                block(
                    self.inplanes,
                    planes,
                    relu_type=self.relu_type,
                )
            )

        return mySequential_resnet(*layers)

    def forward(self, x, udp):
        """forward.

        :param x: torch.Tensor, input tensor with input size (B, C, T, H, W).
        """
        x = self.layer1(x, udp['udp1'].cuda(), udp['udp2'].cuda(), udp['udp3'].cuda(), udp['udp4'].cuda())
        x = self.layer2(x, udp['udp5'].cuda(), udp['udp6'].cuda(), udp['udp7'].cuda(), udp['udp8'].cuda())
        x = self.layer3(x, udp['udp9'].cuda(), udp['udp10'].cuda(), udp['udp11'].cuda(), udp['udp12'].cuda())
        x = self.layer4(x, udp['udp13'].cuda(), udp['udp14'].cuda(), udp['udp15'].cuda(), udp['udp16'].cuda())
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        return x

class mySequential_resnet(nn.Sequential):
    def forward(self, *input):
        x = input[0]
        for i, module in enumerate(self._modules.values()):
            if i == 0:
                x = module(x, *input[1:3])
            else:
                x = module(x, *input[3:5])
        return x