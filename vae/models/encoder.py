import torch
import torch.nn as nn
import torch.nn.functional as F
from .residual import ResidualStack


class Encoder(nn.Module):
    """
    Outputs latent mean and log-variance: μ, logσ²
    """

    def __init__(self, in_dim, h_dim, res_h_dim, n_res_layers, latent_dim):
        super(Encoder, self).__init__()
        self.conv_stack = nn.Sequential(
            nn.Conv2d(in_dim, h_dim // 2, 4, 2, 1),  # [B, h/2, 100, 100]
            nn.ReLU(),
            nn.Conv2d(h_dim // 2, h_dim, 4, 2, 1),  # [B, h, 50, 50]
            nn.ReLU(),
            nn.Conv2d(h_dim, h_dim, 3, 1, 1),  # [B, h, 50, 50]
            ResidualStack(h_dim, h_dim, res_h_dim, n_res_layers)
        )

        dummy_input = torch.zeros(1, in_dim, 200, 200)
        conv_out = self.conv_stack(dummy_input)
        conv_flat_dim = conv_out.view(1, -1).shape[1]

        self.fc_mu = nn.Linear(conv_flat_dim, latent_dim)
        self.fc_logvar = nn.Linear(conv_flat_dim, latent_dim)

    def forward(self, x):
        x = self.conv_stack(x)              # [B, h_dim, 49, 49]
        x = x.view(x.size(0), -1)           # Flatten to [B, h_dim*49*49]
        mu = self.fc_mu(x)                  # [B, latent_dim]
        logvar = self.fc_logvar(x)          # [B, latent_dim]
        return mu, logvar
