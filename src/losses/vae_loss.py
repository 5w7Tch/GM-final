"""
VAE Loss Functions
==================
TODO: Partner's implementation

ELBO = E[log p(x|z)] - KL(q(z|x) || p(z))
"""

import torch
import torch.nn.functional as F
from typing import Tuple


def vae_loss(
    x: torch.Tensor,
    x_recon: torch.Tensor,
    mu: torch.Tensor,
    log_var: torch.Tensor,
    beta: float = 1.0
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    VAE ELBO Loss = Reconstruction + β * KL Divergence

    TODO: Partner's implementation

    Returns: (total_loss, recon_loss, kl_loss)
    """
    # Reconstruction loss (MSE)
    # recon_loss = F.mse_loss(x_recon, x, reduction='sum') / x.size(0)

    # KL divergence: -0.5 * sum(1 + log_var - mu^2 - exp(log_var))
    # kl_loss = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp()) / x.size(0)

    # total_loss = recon_loss + beta * kl_loss

    raise NotImplementedError("Partner's task")


def elbo_loss(
    x: torch.Tensor,
    x_recon: torch.Tensor,
    mu: torch.Tensor,
    log_var: torch.Tensor
) -> torch.Tensor:
    """Standard ELBO loss (beta=1)"""
    total, _, _ = vae_loss(x, x_recon, mu, log_var, beta=1.0)
    return total
