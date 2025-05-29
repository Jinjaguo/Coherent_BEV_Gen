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
        kernel = 4
        stride = 2

        self.conv_stack = nn.Sequential(
            nn.Conv2d(in_dim, h_dim // 2, kernel_size=kernel,
                      stride=stride, padding=1),  # [B, h_dim//2, 100, 100]
            nn.ReLU(),
            nn.Conv2d(h_dim // 2, h_dim, kernel_size=kernel,
                      stride=stride, padding=1),  # [B, h_dim, 50, 50]
            nn.ReLU(),
            nn.Conv2d(h_dim, h_dim, kernel_size=kernel-1,
                      stride=stride-1, padding=1),  # [B, h_dim, 49, 49]
            ResidualStack(h_dim, h_dim, res_h_dim, n_res_layers)
        )

        # 最终 flatten 到全连接层输出 μ 和 logσ²
        conv_output_dim = h_dim * 49 * 49
        self.fc_mu = nn.Linear(conv_output_dim, latent_dim)
        self.fc_logvar = nn.Linear(conv_output_dim, latent_dim)

    def forward(self, x):
        x = self.conv_stack(x)              # [B, h_dim, 49, 49]
        x = x.view(x.size(0), -1)           # Flatten to [B, h_dim*49*49]
        mu = self.fc_mu(x)                  # [B, latent_dim]
        logvar = self.fc_logvar(x)          # [B, latent_dim]
        return mu, logvar
