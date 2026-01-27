"""
Variational Autoencoder (VAE) for CIFAR-10
==========================================
Based on: "Auto-Encoding Variational Bayes" - Kingma & Welling, 2014
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple


class Encoder(nn.Module):
    """
    VAE Encoder: x -> (μ, log σ²)

    """

    def __init__(self, latent_dim: int = 128, image_channels: int = 3):
        super().__init__()
        self.latent_dim = latent_dim
        self.conv = nn.Sequential(
            nn.Conv2d(image_channels, 64, 4, 2, 1),   # 32 -> 16
            nn.ReLU(True),
            nn.Conv2d(64, 128, 4, 2, 1),               # 16 -> 8
            nn.ReLU(True),
            nn.Conv2d(128, 256, 4, 2, 1),              # 8 -> 4
            nn.ReLU(True)
        )

        self.fc_mu = nn.Linear(256 * 4 * 4, latent_dim)
        self.fc_logvar = nn.Linear(256 * 4 * 4, latent_dim)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        h = self.conv(x)
        h = h.view(x.size(0), -1)
        mu = self.fc_mu(h)
        log_var = self.fc_logvar(h)
        return mu, log_var


class Decoder(nn.Module):
    """
    VAE Decoder: z -> x_reconstructed

    """

    def __init__(self, latent_dim: int = 128, image_channels: int = 3):
        super().__init__()
        self.latent_dim = latent_dim

        self.fc = nn.Linear(latent_dim, 256 * 4 * 4)

        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 4, 2, 1),  # 4 -> 8
            nn.ReLU(True),
            nn.ConvTranspose2d(128, 64, 4, 2, 1),   # 8 -> 16
            nn.ReLU(True),
            nn.ConvTranspose2d(64, image_channels, 4, 2, 1),  # 16 -> 32
            nn.Tanh()
        )
    def forward(self, z: torch.Tensor) -> torch.Tensor:
        h = self.fc(z)
        h = h.view(z.size(0), 256, 4, 4)
        return self.deconv(h)

class VAE(nn.Module):
    """
    Full VAE: Encoder + Reparameterization + Decoder

    """

    def __init__(self, latent_dim: int = 128, image_channels: int = 3):
        super().__init__()
        self.latent_dim = latent_dim
        self.encoder = Encoder(latent_dim, image_channels)
        self.decoder = Decoder(latent_dim, image_channels)

    def reparameterize(self, mu: torch.Tensor, log_var: torch.Tensor) -> torch.Tensor:
        """z = μ + σ * ε, where ε ~ N(0, I)"""
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Returns: (x_recon, mu, log_var)"""
        mu, log_var = self.encoder(x)
        z = self.reparameterize(mu, log_var)
        x_recon = self.decoder(z)
        return x_recon, mu, log_var

    @torch.no_grad()
    def sample(self, n_samples: int, device: str = 'cuda') -> torch.Tensor:
        """Sample from prior p(z) = N(0, I)"""
        z = torch.randn(n_samples, self.latent_dim, device=device)
        return self.decoder(z)
