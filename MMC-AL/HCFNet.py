# HCFNet.py
# --------------------------------------------------------
# 论文:HCF-Net: Hierarchical Context Fusion Network for Infrared Small Object Detection (arxiv 2024)
# 论文地址：https://arxiv.org/abs/2403.10778
# ------


import math
import torch
import torch.nn as nn
import torch.nn.functional as F

# 空间注意力模块
class SpatialAttentionModule(nn.Module):
    def __init__(self):
        super(SpatialAttentionModule, self).__init__()
        # 定义一个 2 通道输入的卷积层，输出通道为 1，卷积核大小为 7，步长为 1，填充为 3
        self.conv2d = nn.Conv2d(in_channels=2, out_channels=1, kernel_size=7, stride=1, padding=3)
        # Sigmoid 激活函数
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # 计算平均值
        avgout = torch.mean(x, dim=1, keepdim=True)
        # 计算最大值
        maxout, _ = torch.max(x, dim=1, keepdim=True)
        # 沿通道维度拼接平均值和最大值
        out = torch.cat([avgout, maxout], dim=1)
        # 卷积层 + Sigmoid
        out = self.sigmoid(self.conv2d(out))
        # 返回注意力权重乘以原始输入
        return out * x

# 多级注意力金字塔模块
class PPA(nn.Module):
    def __init__(self, in_features, filters) -> None:
        super().__init__()

        # 1x1 卷积层，用于生成 skip connection
        self.skip = conv_block(in_features=in_features,
                               out_features=filters,
                               kernel_size=(1, 1),
                               padding=(0, 0),
                               norm_type='bn',
                               activation=False)
        # 第一个 3x3 卷积层
        self.c1 = conv_block(in_features=in_features,
                             out_features=filters,
                             kernel_size=(3, 3),
                             padding=(1, 1),
                             norm_type='bn',
                             activation=True)
        # 第二个 3x3 卷积层
        self.c2 = conv_block(in_features=filters,
                             out_features=filters,
                             kernel_size=(3, 3),
                             padding=(1, 1),
                             norm_type='bn',
                             activation=True)
        # 第三个 3x3 卷积层
        self.c3 = conv_block(in_features=filters,
                             out_features=filters,
                             kernel_size=(3, 3),
                             padding=(1, 1),
                             norm_type='bn',
                             activation=True)
        # 空间注意力模块
        self.sa = SpatialAttentionModule()
        # ECA 注意力模块
        self.cn = ECA(filters)
        # 2x2 局部全局注意力模块
        self.lga2 = LocalGlobalAttention(filters, 2)
        # 4x4 局部全局注意力模块
        self.lga4 = LocalGlobalAttention(filters, 4)

        # 批量归一化层
        self.bn1 = nn.BatchNorm2d(filters)
        # Dropout2d 层
        self.drop = nn.Dropout2d(0.1)
        # ReLU 激活函数
        self.relu = nn.ReLU()
        # GELU 激活函数
        self.gelu = nn.GELU()

    def forward(self, x):
        # 生成 skip connection
        x_skip = self.skip(x)
        # 2x2 局部全局注意力
        x_lga2 = self.lga2(x_skip)
        # 4x4 局部全局注意力
        x_lga4 = self.lga4(x_skip)
        # 第一个 3x3 卷积
        x1 = self.c1(x)
        # 第二个 3x3 卷积
        x2 = self.c2(x1)
        # 第三个 3x3 卷积
        x3 = self.c3(x2)
        # 各个分支结果相加
        x = x1 + x2 + x3 + x_skip + x_lga2 + x_lga4
        # ECA 注意力
        x = self.cn(x)
        # 空间注意力
        x = self.sa(x)
        # Dropout2d
        x = self.drop(x)
        # 批量归一化
        x = self.bn1(x)
        # ReLU 激活
        x = self.relu(x)
        # 返回最终输出
        return x

# 局部全局注意力模块
class LocalGlobalAttention(nn.Module):
    def __init__(self, output_dim, patch_size):
        super().__init__()
        self.output_dim = output_dim
        self.patch_size = patch_size
        # 线性层
        self.mlp1 = nn.Linear(patch_size * patch_size, output_dim // 2)
        # 层归一化
        self.norm = nn.LayerNorm(output_dim // 2)
        # 线性层
        self.mlp2 = nn.Linear(output_dim // 2, output_dim)
        # 1x1 卷积层
        self.conv = nn.Conv2d(output_dim, output_dim, kernel_size=1)
        # 可学习的提示向量
        self.prompt = torch.nn.parameter.Parameter(torch.randn(output_dim, requires_grad=True))
        # 可学习的变换矩阵
        self.top_down_transform = torch.nn.parameter.Parameter(torch.eye(output_dim), requires_grad=True)

    def forward(self, x):
        # 调整维度顺序
        x = x.permute(0, 2, 3, 1)
        B, H, W, C = x.shape
        P = self.patch_size

        # 局部分支
        # 局部补丁提取
        local_patches = x.unfold(1, P, P).unfold(2, P, P)  # (B, H/P, W/P, P, P, C)
        # 调整形状
        local_patches = local_patches.reshape(B, -1, P * P, C)  # (B, H/P*W/P, P*P, C)
        # 计算每个局部补丁的平均值
        local_patches = local_patches.mean(dim=-1)  # (B, H/P*W/P, P*P)

        # MLP 层
        local_patches = self.mlp1(local_patches)  # (B, H/P*W/P, input_dim // 2)
        # 层归一化
        local_patches = self.norm(local_patches)  # (B, H/P*W/P, input_dim // 2)
        # MLP 层
        local_patches = self.mlp2(local_patches)  # (B, H/P*W/P, output_dim)

        # Softmax 激活
        local_attention = F.softmax(local_patches, dim=-1)  # (B, H/P*W/P, output_dim)
        # 注意力加权
        local_out = local_patches * local_attention  # (B, H/P*W/P, output_dim)

        # Cosine 相似性计算
        cos_sim = F.normalize(local_out, dim=-1) @ F.normalize(self.prompt[None, ..., None], dim=1)  # B, N, 1
        # 掩码
        mask = cos_sim.clamp(0, 1)
        # 应用掩码
        local_out = local_out * mask
        # 应用变换
        local_out = local_out @ self.top_down_transform

        # 恢复形状
        local_out = local_out.reshape(B, H // P, W // P, self.output_dim)  # (B, H/P, W/P, output_dim)
        # 调整维度顺序
        local_out = local_out.permute(0, 3, 1, 2)
        # 上采样
        local_out = F.interpolate(local_out, size=(H, W), mode='bilinear', align_corners=False)
        # 1x1 卷积
        output = self.conv(local_out)

        # 返回最终输出
        return output

# ECA 注意力模块
class ECA(nn.Module):
    def __init__(self, in_channel, gamma=2, b=1):
        super(ECA, self).__init__()
        # 计算卷积核大小
        k = int(abs((math.log(in_channel, 2) + b) / gamma))
        kernel_size = k if k % 2 else k + 1
        padding = kernel_size // 2
        # 平均池化
        self.pool = nn.AdaptiveAvgPool2d(output_size=1)
        # 1D 卷积层 + Sigmoid
        self.conv = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=1, kernel_size=kernel_size, padding=padding, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        # 平均池化
        out = self.pool(x)
        # 调整维度
        out = out.view(x.size(0), 1, x.size(1))
        # 1D 卷积 + Sigmoid
        out = self.conv(out)
        # 调整维度
        out = out.view(x.size(0), x.size(1), 1, 1)
        # 返回注意力权重乘以原始输入
        return out * x

# 卷积块
class conv_block(nn.Module):
    def __init__(self,
                 in_features,
                 out_features,
                 kernel_size=(3, 3),
                 stride=(1, 1),
                 padding=(1, 1),
                 dilation=(1, 1),
                 norm_type='bn',
                 activation=True,
                 use_bias=True,
                 groups=1
                 ):
        super().__init__()
        # 定义卷积层
        self.conv = nn.Conv2d(in_channels=in_features,
                              out_channels=out_features,
                              kernel_size=kernel_size,
                              stride=stride,
                              padding=padding,
                              dilation=dilation,
                              bias=use_bias,
                              groups=groups)

        self.norm_type = norm_type
        self.act = activation

        # 根据归一化类型选择不同的归一化层
        if self.norm_type == 'gn':
            self.norm = nn.GroupNorm(32 if out_features >= 32 else out_features, out_features)
        elif self.norm_type == 'bn':
            self.norm = nn.BatchNorm2d(out_features)
        # 激活函数
        if self.act:
            # self.relu = nn.GELU()
            self.relu = nn.ReLU(inplace=False)

    def forward(self, x):
        # 卷积层
        x = self.conv(x)
        # 归一化层
        if self.norm_type is not None:
            x = self.norm(x)
        # 激活函数
        if self.act:
            x = self.relu(x)
        # 返回输出
        return x

# 主程序入口
if __name__ == '__main__':
    # 创建 PPA 模块实例
    block = PPA(in_features=64, filters=64)  # in_features：输入通道数，filters：输出通道数
    # 创建随机输入
    input = torch.rand(32, 64, 64, 64)
    # 前向传播
    output = block(input)
    # 打印输入和输出的大小
    print(input.size())
    print(output.size())
    print("抖音、B站、小红书、CSDN同号")
    print("布尔大学士 提醒您：代码无误~~~~")