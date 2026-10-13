"""Paper-consistent NC-CAL generator for ShanghaiTech."""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .final_future_prediction_ped2 import Encoder as MaskEncoder
from .final_future_prediction_ped2 import OffsetNet as SharedOffsetNet
from .final_future_prediction_ped2 import Smooth_Loss, Test_Loss
from .unet import UNet_SSPCAB


class convAE_me(nn.Module):
    """U-Net + 2,000-item memory + SSPCAB + PDM used in the paper."""

    def __init__(self, n_channel=3, t_length=5, bkg=None):
        super().__init__()
        if n_channel != 3:
            raise ValueError('NC-CAL expects four RGB input frames (12 channels)')

        self.unet_SSPCAB = UNet_SSPCAB(input_channels=12, output_channel=3)
        self.encoder = MaskEncoder(t_length, n_channel)
        self.offset_net = SharedOffsetNet(t_length, n_channel)

        if bkg is None:
            background = torch.zeros((437, 3, 256, 256), dtype=torch.float32)
        else:
            background = torch.from_numpy(
                bkg.transpose((0, 3, 1, 2)).astype(np.float32) / 127.5 - 1
            )
        # Background templates are fixed data inputs, not learnable parameters.
        self.register_buffer('bkg', background)

        self.mse1 = 0
        self.mse2 = 0
        self.mse3 = 0
        self.loss_sparsity = 0
        self.smooth_loss = 0
        self.offset_loss1 = 0
        self.offset_loss2 = 0
        self.err1 = 0
        self.err2_ = 0

        self.loss_smooth2 = Smooth_Loss(ks=3)
        self.loss_smooth1 = Smooth_Loss(ks=7)
        self.get_err = Test_Loss(ks=(6, 18))
        self.get_err_ = Test_Loss(3, ks=(42, 31))

    def forward(self, x, idx, test=False):
        del test  # Kept in the public interface for evaluation compatibility.
        grid1, grid2, offset1, offset2 = self.offset_net(x)
        _, _, _, mask = self.encoder(x)
        z_q, att, sspcab_input, sspcab_output = self.unet_SSPCAB(x)

        z_q_ = F.grid_sample(z_q, grid1, align_corners=True)
        z_q_t = F.grid_sample(z_q_, grid2, align_corners=True)
        mask1 = F.grid_sample(mask, grid1, align_corners=True)
        mask2 = F.grid_sample(mask1, grid2, align_corners=True)

        background = self.bkg[idx]
        z_q_ = z_q_ * mask1 + (1 - mask1) * background
        z_q_t = z_q_t * mask2 + (1 - mask2) * background

        return (
            z_q_t,
            z_q_,
            z_q,
            offset1,
            offset2,
            mask,
            mask1,
            mask2,
            sspcab_input,
            sspcab_output,
            att,
        )

    def loss_function(
        self,
        x,
        idx,
        recon_x,
        z_q_,
        z_q,
        offset1=None,
        offset2=None,
        mask=None,
        mask1=None,
        mask2=None,
        SSPCAB_input=None,
        SSPCAB_output=None,
        att=None,
        compute_err=False,
    ):
        del z_q, mask, mask1, mask2
        if compute_err:
            pixel_error = F.mse_loss(recon_x, x, reduction='none')
            self.err1 = self.get_err((offset1 ** 2).sum(-1) ** 0.5)
            self.err2_ = self.get_err_(pixel_error)
            return pixel_error.mean(1)

        # ShanghaiTech uses the paper's background-weighted two-output MSE.
        weight = 1 + 100 * torch.abs(
            F.avg_pool2d(
                (x - self.bkg[idx]).abs() ** 0.5,
                kernel_size=15,
                stride=1,
                padding=7,
            )
        )
        self.mse1 = (F.mse_loss(z_q_, x, reduction='none') * weight).sum() / weight.sum()
        self.mse2 = (F.mse_loss(recon_x, x, reduction='none') * weight).sum() / weight.sum()
        self.mse3 = F.mse_loss(SSPCAB_input, SSPCAB_output)
        self.loss_sparsity = torch.mean(
            torch.sum(-att * torch.log(att + 1e-12), dim=1)
        )
        self.offset_loss1 = ((offset1 ** 2).sum(dim=-1) ** 0.5).mean()
        self.offset_loss2 = ((offset2 ** 2).sum(dim=-1) ** 0.5).mean()
        self.smooth_loss = (
            self.loss_smooth1(offset1.permute(0, 3, 1, 2))
            + self.loss_smooth2(offset2.permute(0, 3, 1, 2))
        )

        return (
            self.mse1
            + self.mse2
            + self.mse3
            + 0.002 * self.loss_sparsity
            + self.offset_loss1
            + self.offset_loss2
            + self.smooth_loss
        )

    def latest_losses(self):
        return {
            'mse1': self.mse1,
            'mse2': self.mse2,
            'sspcab': self.mse3,
            'sparsity': self.loss_sparsity,
            'offset1': self.offset_loss1,
            'offset2': self.offset_loss2,
            'smooth': self.smooth_loss,
            'err1': self.err1,
            'err2_': self.err2_,
        }

    def load_bkg(self, bkg):
        background = torch.from_numpy(
            bkg.transpose((0, 3, 1, 2)).astype(np.float32) / 127.5 - 1
        )
        self.bkg = background.to(self.bkg.device)
