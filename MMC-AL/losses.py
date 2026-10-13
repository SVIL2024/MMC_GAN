"""Adversarial objectives used by the paper's two-phase training protocol."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class Adversarial_Loss(nn.Module):
    """Non-saturating generator loss: -E[log D(G(x))]."""

    def forward(self, fake_outputs):
        return F.binary_cross_entropy(fake_outputs, torch.ones_like(fake_outputs))


class Discriminate_Loss(nn.Module):
    """Binary cross-entropy discriminator loss from the paper."""

    def forward(self, real_outputs, fake_outputs):
        real_loss = F.binary_cross_entropy(real_outputs, torch.ones_like(real_outputs))
        fake_loss = F.binary_cross_entropy(fake_outputs, torch.zeros_like(fake_outputs))
        return real_loss + fake_loss
