import torch
import torch.nn as nn
from torch.nn import functional as F
class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16): # in_planes是输入通道数
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1) # 全局自适应平均池化
        self.max_pool = nn.AdaptiveMaxPool2d(1) # 全局自适应最大池化
        # nn.Conv2d的参数依次是输入通道数、输出通道数、卷积核尺寸
        # 所以fc1是将输入通道数降为原来的十六分之一，
        self.fc1   = nn.Conv2d(in_planes, in_planes // 16, 1, bias=False)
        self.relu1 = nn.ReLU() # ReLU激活函数
        # fc2则又实现升维度
        self.fc2   = nn.Conv2d(in_planes // 16, in_planes, 1, bias=False)

        self.sigmoid = nn.Sigmoid() # Sigmoid激活函数

    def forward(self, x): # 前向传播
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        # 两种结果相加之后再经过sigmoid函数激活
        return self.sigmoid(out)
# 空间注意力模块
class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1 # 如果卷积核大小为7X7则填充三行三列，是3则填充1行1列

        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True) # 计算通道维度的平均值。
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1) # 连接上面两个通道得到双通道
        x = self.conv1(x) # 对双通道进行卷积输出单通道
        return self.sigmoid(x) # 再对输出的单通道激活