import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import TransformerEncoderLayer
# import sspcab_torch
from .sspcab_torch import SSPCAB
from .mem import MemModule
# from sspcab_torch import SSPCAB
# from models.final_future_prediction_ped2 import Get_Mask,OffsetNet
class double_conv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Sequential(nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
                                  nn.BatchNorm2d(out_ch),
                                  nn.ReLU(inplace=True),
                                  nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
                                  nn.BatchNorm2d(out_ch),
                                  nn.ReLU(inplace=True))

    def forward(self, x):
        x = self.conv(x)
        return x


class inconv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = double_conv(in_ch, out_ch)

    def forward(self, x):
        x = self.conv(x)
        return x


class down(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.mpconv = nn.Sequential(nn.MaxPool2d(2),
                                    double_conv(in_ch, out_ch))

    def forward(self, x):
        x = self.mpconv(x)
        return x


class up(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_ch, in_ch // 2, 2, stride=2)
        self.conv = double_conv(in_ch, out_ch)

    def forward(self, x1, x2):
        x1 = self.up(x1)

        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, (diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2))
        # for padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd

        x = torch.cat([x2, x1], dim=1)
        x = self.conv(x)
        return x

#如果mem等于True则使用记忆模块，False则不使用记忆模块
mem = True

class UNet(nn.Module):
    def __init__(self, input_channels, output_channel=3):
        super(UNet, self).__init__()
        self.inc = inconv(input_channels, 64)
        self.down1 = down(64, 128)
        self.down2 = down(128, 256)
        self.down3 = down(256, 512)


        #1加入记忆模块,这里的参数给定死了，不然的话，可以通过unet的构造器进行传参，一行代码
        if mem:
            self.mem_module1 = MemModule(mem_dim=2000, fea_dim=512, shrink_thres=0.0025)

        self.up1 = up(512, 256)
        self.up2 = up(256, 128)
        self.up3 = up(128, 64)
        self.outc = nn.Conv2d(64, output_channel, kernel_size=3, padding=1)

    def forward(self, x):

        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)

        #1这里加入了记忆模块:两行代码
        if mem:
            x4 = self.mem_module1(x4)
            att = x4['att']
            x4 = x4['output']
        x = self.up1(x4, x3)
        x = self.up2(x, x2)
        x = self.up3(x, x1)
        x = self.outc(x)
        output = torch.tanh(x)

        return output, att if mem else output


class UNet_SSPCAB(nn.Module):
    def __init__(self, input_channels, output_channel=3):
        super(UNet_SSPCAB, self).__init__()
        self.inc = inconv(input_channels, 64)
        self.down1 = down(64, 128)
        self.down2 = down(128, 256)
        self.down3 = down(256, 512)

        #1加入记忆模块,这里的参数给定死了，不然的话，可以通过unet的构造器进行传参，一行代码
        if mem:
            self.mem_module1 = MemModule(mem_dim=2000, fea_dim=512, shrink_thres=0.0025)
        self.up1 = up(512, 256)
        self.up2 = up(256, 128)
        self.up3 = up(128, 64)
        # 插入 SSPCAB 模块
        self.sspcab = SSPCAB(channels=64)
        self.outc = nn.Conv2d(64, output_channel, kernel_size=3, padding=1)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)

        #1这里加入了记忆模块:两行代码
        if mem:
            x4 = self.mem_module1(x4)
            #想办法把这个x4里面的att属性返回出去，方便下面做熵损失
            att = x4['att']
            x4 = x4['output']

        x = self.up1(x4, x3)
        x = self.up2(x, x2)

        SSPCAB_input = self.up3(x, x1)
        SSPCAB_output = self.sspcab(SSPCAB_input)
        x = self.outc(SSPCAB_output)
        output = torch.tanh(x)

        return output, att,SSPCAB_input , SSPCAB_output if mem else output

class UNet_transformerOne(nn.Module):
    def __init__(self, input_channels, output_channel=3):
        super(UNet_transformerOne, self).__init__()
        self.inc = inconv(input_channels, 64)
        self.down1 = down(64, 128)
        self.down2 = down(128, 256)
        self.down3 = down(256, 512)

        if mem:
            self.mem_module1 = MemModule(mem_dim=2000, fea_dim=512, shrink_thres=0.0025)

        self.up1 = up(512, 256)
        self.up2 = up(256, 128)
        self.up3 = up(128, 64)
        self.outc = nn.Conv2d(64, output_channel, kernel_size=3, padding=1)

        # Transformer Encoder
        self.transformer = TransformerEncoderLayer(d_model=512, nhead=8, dim_feedforward=2048, dropout=0.1)
        self.transformer_encoder = nn.TransformerEncoder(self.transformer, num_layers=1)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)

        if mem:
            x4 = self.mem_module1(x4)
            att = x4['att']
            x4 = x4['output']

        # Flatten the feature map for Transformer
        B, C, H, W = x4.size()
        x4 = x4.view(B, C, -1).transpose(1, 2)

        # Apply Transformer
        x4 = self.transformer_encoder(x4)

        # Reshape back to original feature map shape
        x4 = x4.transpose(1, 2).view(B, C, H, W)

        x = self.up1(x4, x3)
        x = self.up2(x, x2)
        x = self.up3(x, x1)
        x = self.outc(x)
        output = torch.tanh(x)

        return output, att if mem else output


class UNet_transformerAll(nn.Module):
    def __init__(self, input_channels, output_channel=3):
        super(UNet_transformerAll, self).__init__()
        self.inc = inconv(input_channels, 64)
        self.down1 = down(64, 128)
        self.down2 = down(128, 256)
        self.down3 = down(256, 512)

        if mem:
            self.mem_module1 = MemModule(mem_dim=2000, fea_dim=512, shrink_thres=0.0025)

        self.up1 = up(512, 256)
        self.up2 = up(256, 128)
        self.up3 = up(128, 64)
        self.outc = nn.Conv2d(64, output_channel, kernel_size=3, padding=1)

        # Transformer Encoder
        self.transformer = TransformerEncoderLayer(d_model=512, nhead=8, dim_feedforward=2048, dropout=0.1)
        self.transformer_encoder = nn.TransformerEncoder(self.transformer, num_layers=1)

    def forward(self, x):
        x1 = self.inc(x)
        x1 = self.transformer_encoder(x1)

        x2 = self.down1(x1)
        x2 = self.transformer_encoder(x2)

        x3 = self.down2(x2)
        x3 = self.transformer_encoder(x3)

        x4 = self.down3(x3)
        if mem:
            x4 = self.mem_module1(x4)
            att = x4['att']
            x4 = x4['output']
        x4 = self.transformer_encoder(x4)

        x = self.up1(x4, x3)
        x = self.up2(x, x2)
        x = self.up3(x, x1)
        x = self.outc(x)
        output = torch.tanh(x)
        return output, att if mem else output


def _test():
    rand = torch.ones([4, 12, 256, 256]).cuda()
    t = UNet(12, 3).cuda()
    r = t(rand)
    print(r.shape)
    print(r.grad_fn)
    print(r.requires_grad)

if __name__ == '__main__':
    rand = torch.ones([1, 12, 256, 256]).cuda()
    t= UNet(12, 3).cuda()


    r = t(rand)





    print(r.shape)
    print(r.grad_fn)
    print(r.requires_grad)


