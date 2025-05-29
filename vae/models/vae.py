import torch
import torch.nn as nn
import torch.nn.functional as F
from .encoder import Encoder
from .decoder import Decoder


class VAE(nn.Module):
    def __init__(self, h_dim, res_h_dim, n_res_layers, latent_dim):
        super(VAE, self).__init__()
        self.encoder = Encoder(1, h_dim, res_h_dim, n_res_layers, latent_dim)
        self.decoder = Decoder(latent_dim, h_dim, n_res_layers, res_h_dim)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x, verbose=False):
        mu, logvar = self.encoder(x)
        z = self.reparameterize(mu, logvar)
        x_hat = self.decoder(z)

        if verbose:
            print('original data shape:', x.shape)
            print('reconstructed shape:', x_hat.shape)
            print('latent mu shape:', mu.shape)
            print('latent logvar shape:', logvar.shape)
            assert False

        return x_hat, mu, logvar



