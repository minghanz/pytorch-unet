# Adapted from https://discuss.pytorch.org/t/unet-implementation/426

import torch
from torch import nn
import torch.nn.functional as F


class UNetRegressor(nn.Module):
    def __init__(
        self,
        feature_channels=64,
        wf=3,
        padding=False,
        batch_norm=False,
    ):
        """
        Implementation of
        U-Net: Convolutional Networks for Biomedical Image Segmentation
        (Ronneberger et al., 2015)
        https://arxiv.org/abs/1505.04597

        Using the default arguments will yield the exact version used
        in the original paper

        Args:
            in_channels (int): number of input channels
            n_classes (int): number of output channels
            depth (int): depth of the network
            wf (int): number of filters in the first layer is 2**wf
            padding (bool): if True, apply padding such that the input shape
                            is the same as the output.
                            This may introduce artifacts
            batch_norm (bool): Use BatchNorm after layers with an
                               activation function
            up_mode (str): one of 'upconv' or 'upsample'.
                           'upconv' will use transposed convolutions for
                           learned upsampling.
                           'upsample' will use bilinear upsampling.
        """
        super(UNetRegressor, self).__init__()
        self.padding = padding
        self.down_path_feature_map = nn.ModuleList()
        self.down_path_depth_map = nn.ModuleList()

        prev_channels_feature = feature_channels # 64
        next_channels_feature = prev_channels_feature / 2 # 32
        prev_channels_depth = 1
        next_channels_depth = 2 **wf # 8
        self.down_path_feature_map.append(
            UNetConvBlock(prev_channels_feature, next_channels_feature, padding, batch_norm)
        )
        self.down_path_depth_map.append(
            UNetConvBlock(prev_channels_depth, next_channels_depth, padding, batch_norm)
        )

        prev_channels_feature = next_channels_feature # 32
        next_channels_feature = prev_channels_feature / 2 # 16
        prev_channels_depth = next_channels_depth # 8
        next_channels_depth = prev_channels_depth * 2 # 16
        self.down_path_feature_map.append(
            UNetConvBlock(prev_channels_feature + prev_channels_depth, next_channels_feature, padding, batch_norm)
        )
        self.down_path_depth_map.append(
            UNetConvBlock(prev_channels_depth, next_channels_depth, padding, batch_norm)
        )

        prev_channels_feature = next_channels_feature # 16
        next_channels_feature = prev_channels_feature # 16
        prev_channels_depth = next_channels_depth # 16
        self.down_path_feature_map.append(
            UNetConvBlock(prev_channels_feature + prev_channels_depth, next_channels_feature, padding, batch_norm)
        )

        self.regressor_block = RegressionBlock(prev_channels_feature, 6)


    def forward(self, feature_map_1, feature_map_2, depth_map_1, depth_map_2):
        feature_map = torch.cat([feature_map_1, feature_map_2], 1)
        depth_map = torch.cat([depth_map_1, depth_map_2], 1)
        
        for i in range(2):
            feature_map = self.down_path_feature_map[i](feature_map)
            depth_map = self.down_path_depth_map[i](depth_map)
            feature_map = torch.cat([feature_map, depth_map], 1)
        feature_map = self.down_path_feature_map[i+1](feature_map)

        prediction = self.regressor_block(feature_map)
        prediction = torch.squeeze(prediction)
        return prediction

class UNetConvBlock(nn.Module):
    def __init__(self, in_size, out_size, padding, batch_norm):
        super(UNetConvBlock, self).__init__()
        block = []

        block.append(nn.Conv2d(in_size, out_size, kernel_size=3, padding=int(padding)))
        block.append(nn.ReLU())
        if batch_norm:
            block.append(nn.BatchNorm2d(out_size))

        block.append(nn.Conv2d(out_size, out_size, kernel_size=3, padding=int(padding)))
        block.append(nn.ReLU())
        if batch_norm:
            block.append(nn.BatchNorm2d(out_size))

        self.block = nn.Sequential(*block)

    def forward(self, x):
        out = self.block(x)
        return out


class RegressionBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(RegressionBlock, self).__init__()
        block = []

        block.append(nn.Conv2d(in_channels, out_channels, kernel_size=3) )
        block.append(nn.AdaptiveAvgPool2d(1))

        self.block = nn.Sequential(*block)


    def forward(self, x):
        out = self.block(x)
        return out
