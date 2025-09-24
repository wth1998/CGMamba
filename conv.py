import torch
import torch.nn as nn


class ConvMultiScaleFeatureExtractor(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ConvMultiScaleFeatureExtractor, self).__init__()
        self.conv1x1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )
        self.conv3x3 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )
        self.conv5x5 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=5, padding=2),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )
        # 特征融合
        self.fusion_conv = nn.Sequential(
            nn.Conv2d(out_channels * 3, out_channels, kernel_size=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )
        # # 残差连接
        # self.residual = nn.Identity()

    def forward(self, x):
        # 多尺度特征提取
        x1 = self.conv1x1(x)
        x2 = self.conv3x3(x)
        x3 = self.conv5x5(x)
        # 特征拼接
        x_concat = torch.cat([x1, x2, x3], dim=1)
        # 特征融合
        x_concat = self.fusion_conv(x_concat)
        # 残差连接
        # x += self.residual(x)
        x = x + x_concat
        return x
