import torch
import torch.nn as nn
import torch.nn.functional as F

# 宽卷积层
class WideConvLayer(nn.Module):
    def __init__(self, in_channels):
        super(WideConvLayer, self).__init__()
        # 两个不同卷积核大小的卷积层
        self.conv1 = nn.Conv1d(in_channels, 64, kernel_size=(128,), stride=2, padding=63)  # 卷积核大小 128x1，填充 63 保证输入输出尺寸相同
        self.conv2 = nn.Conv1d(in_channels, 64, kernel_size=(64,), stride=2, padding=31)  # 卷积核大小 64x1，填充 31 保证输入输出尺寸相同

    def forward(self, x):
        x1 = F.relu(self.conv1(x))
        x2 = F.relu(self.conv2(x))
        x = torch.cat((x1, x2), dim=1)  # 在通道维度拼接
        return x

# MSASCblock 多尺度自适应卷积块
class MSASCblock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(MSASCblock, self).__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=(5,), stride=2, padding=2)
        self.conv2 = nn.Conv1d(in_channels, out_channels, kernel_size=(5,), stride=2, padding=2)
        self.conv3 = nn.Conv1d(in_channels, out_channels, kernel_size=(5,), stride=2, padding=2)

    def forward(self, x):
        x1 = F.relu(self.conv1(x))
        x2 = F.relu(self.conv2(x))
        x3 = F.relu(self.conv3(x))
        x = torch.cat((x1, x2, x3), dim=1)  # 在通道维度拼接
        return x

# 主模型
class MSASCnn(nn.Module):
    def __init__(self, in_channels, num_classes):
        super(MSASCnn, self).__init__()
        self.wide_conv = WideConvLayer(in_channels)
        self.pool1 = nn.MaxPool1d(kernel_size=(2,), stride=2)
        self.msasc_block1 = MSASCblock(128, 128)  # 修改输入通道数为 128
        self.msasc_block2 = MSASCblock(384, 256)  # 修改输入通道数为 384
        self.pool2 = nn.MaxPool1d(kernel_size=(2,), stride=2)
        self.msasc_block3 = MSASCblock(768, 512)  # 修改输入通道数为 768
        self.global_avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(1536, num_classes)  # 修改输入维度为 1536

    def forward(self, x):
        x = self.wide_conv(x)
        x = self.pool1(x)
        x = self.msasc_block1(x)
        x = self.msasc_block2(x)
        x = self.pool2(x)
        x = self.msasc_block3(x)
        x = self.global_avg_pool(x)
        x = x.view(x.size(0), -1)  # 展平
        x = self.fc(x)
        return x