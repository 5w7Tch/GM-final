"""
Variational Autoencoder (VAE) for CIFAR-10
==========================================
TODO: Partner's implementation

Based on: "Auto-Encoding Variational Bayes" - Kingma & Welling, 2014
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple


class Encoder(nn.Module):
    """
    VAE Encoder: x -> (μ, log σ²)

    TODO: Implement convolutional encoder
    """

    def __init__(self, latent_dim: int = 128, image_channels: int = 3):
        super().__init__()
        self.latent_dim = latent_dim

        # TODO: Implement layers
        raise NotImplementedError("Partner's task")

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        raise NotImplementedError("Partner's task")


class Decoder(nn.Module):
    """
    VAE Decoder: z -> x_reconstructed

    TODO: Implement convolutional decoder
    """

    def __init__(self, latent_dim: int = 128, image_channels: int = 3):
        super().__init__()
        self.latent_dim = latent_dim

        # TODO: Implement layers
        raise NotImplementedError("Partner's task")

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError("Partner's task")


class VAE(nn.Module):
    """
    Full VAE: Encoder + Reparameterization + Decoder

    TODO: Implement complete VAE
    """

    def __init__(self, latent_dim: int = 128, image_channels: int = 3):
        super().__init__()
        self.latent_dim = latent_dim
        self.encoder = Encoder(latent_dim, image_channels)
        self.decoder = Decoder(latent_dim, image_channels)

    def reparameterize(self, mu: torch.Tensor, log_var: torch.Tensor) -> torch.Tensor:
        """z = μ + σ * ε, where ε ~ N(0, I)"""
        # TODO: Implement
        raise NotImplementedError("Partner's task")

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Returns: (x_recon, mu, log_var)"""
        raise NotImplementedError("Partner's task")

    @torch.no_grad()
    def sample(self, n_samples: int, device: str = 'cuda') -> torch.Tensor:
        """Sample from prior p(z) = N(0, I)"""
        z = torch.randn(n_samples, self.latent_dim, device=device)
        return self.decoder(z)
