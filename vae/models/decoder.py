import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from .residual import ResidualStack


class Decoder(nn.Module):
    """
    This is the p_phi (x|z) network. Given a latent sample z, p_phi
    maps back to the original space x.

    - latent_dim: size of latent vector z
    - h_dim: hidden channels used in conv layers
    - res_h_dim: hidden dim inside residual blocks
    - n_res_layers: number of residual layers
    """

    def __init__(self, latent_dim, h_dim, n_res_layers, res_h_dim):
        super(Decoder, self).__init__()
        self.latent_dim = latent_dim
        self.h_dim = h_dim
        self.init_hw = 49  # spatial size to match encoder output

        self.fc = nn.Linear(latent_dim, h_dim * self.init_hw * self.init_hw)

        self.inverse_conv_stack = nn.Sequential(
            nn.ConvTranspose2d(h_dim, h_dim, kernel_size=3, stride=1, padding=1),
            ResidualStack(h_dim, h_dim, res_h_dim, n_res_layers),
            nn.ConvTranspose2d(h_dim, h_dim // 2, kernel_size=4, stride=2, padding=1),  # [49 → 98]
            nn.ReLU(),
            nn.ConvTranspose2d(h_dim // 2, 1, kernel_size=4, stride=2, padding=1),      # [98 → 196]
            nn.Upsample(size=(200, 200), mode='bilinear', align_corners=False)         # final fix
        )

    def forward(self, z):
        x = self.fc(z)                                  # [B, h_dim*49*49]
        x = x.view(-1, self.h_dim, self.init_hw, self.init_hw)  # [B, h_dim, 49, 49]
        x = self.inverse_conv_stack(x)                  # [B, 3, 200, 200]
        return torch.sigmoid(x)