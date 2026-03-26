import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Union
import numpy as np
import math

def get_norm_layer(norm_type, num_features):
    if norm_type == 'batch_norm':
        return nn.BatchNorm2d(num_features)
    elif norm_type == 'layer_norm':
        return LayerNorm(num_features, eps=1e-6)
    else:
        raise ValueError(f"Unsupported norm type: {norm_type}")

class LayerNorm(nn.Module):
    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_first"):
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

class Conv2D(nn.Module):
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: Union[int, tuple],
                 strides: int = 1,
                 padding: Union[int, str] = 'same',
                 dilation: int = 1,
                 groups: int = 1,
                 activation: nn = nn.GELU,
                 dropout_rate: float = 0.0,
                 if_act: bool = True,
                 if_batch_norm: bool = True):
        super(Conv2D, self).__init__()
        layers = []
        bias = False if if_batch_norm else True
        layers.append(nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=strides, padding=padding, dilation=dilation, groups=groups, bias=bias))
        if(if_act):
            layers.append(activation())
        if(if_batch_norm):
            layers.append(nn.BatchNorm2d(out_channels))
        if(dropout_rate>0.0):
            layers.append(nn.Dropout2d(dropout_rate))
        self.convolution2D = nn.Sequential(*layers)

    def forward(self, x):
        return self.convolution2D(x)

class ConvStack(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, groups):
        super(ConvStack, self).__init__()
        current_rf = 1  # Начальное значение рецептивного поля
        layers = []
        # Добавляем слои с приоритетом 7x7
        while current_rf < kernel_size:

            # Если не хватает ровно 2 пикселей для достижения целевого рецептивного поля, добавляем 3x3
            if kernel_size - current_rf == 2:
                layers.append(nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, groups=groups))
                current_rf += 2  # Увеличиваем рецептивное поле на 2

            # Если не хватает ровно 4 пикселей, добавляем 5x5
            elif kernel_size - current_rf == 4:
                layers.append(nn.Conv2d(in_channels, out_channels, kernel_size=5, padding=2, groups=groups))
                current_rf += 4  # Увеличиваем рецептивное поле на 4

            # В остальных случаях добавляем 7x7
            else:
                layers.append(nn.Conv2d(in_channels, out_channels, kernel_size=7, padding=3, groups=groups))
                current_rf += 6  # Увеличиваем рецептивное поле на 6

        layers.sort(key=lambda x: x.kernel_size[0])
        self.conv_stack = nn.Sequential(*layers)

    def forward(self, x):
        return self.conv_stack(x)

class FactorizedConv2D(nn.Module):
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: Union[int, tuple],
                 strides: int = 1,
                 dilation: int = 1,
                 groups: int = 1):
        super(FactorizedConv2D, self).__init__()
        self.conv_1 = nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=(1, kernel_size[1]), stride=strides, padding=(0, kernel_size[1] // 2), dilation=dilation, groups=groups)
        self.conv_2 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=(kernel_size[0], 1), stride=strides, padding=(kernel_size[0] // 2, 0), dilation=dilation, groups=groups)

    def forward(self, x):
        x = self.conv_1(x)
        x = self.conv_2(x)
        return x

class EfficientChannelAttention(nn.Module):
    def __init__(self, kernel_size=3):
        super(EfficientChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=kernel_size, padding=kernel_size // 2, bias=False)

    def forward(self, x):
        # feature descriptor on the global spatial information
        y = self.avg_pool(x).view(x.size(0), 1, x.size(1))

        # Two different branches of ECA module
        y = self.conv(y)

        # Multi-scale information fusion
        y = torch.sigmoid(y).view(x.size(0), x.size(1), 1, 1)

        return x * y.expand_as(x)