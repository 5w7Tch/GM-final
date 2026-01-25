"""
Denoising Score Matching Loss
=============================
For training NCSN to estimate ∇_x log p(x)
"""

import torch
import torch.nn as nn


def dsm_loss(
    score_net: nn.Module,
    x: torch.Tensor,
    sigma: torch.Tensor,
    labels: torch.Tensor
) -> torch.Tensor:
    """
    Denoising Score Matching loss for a single noise level.

    Loss = E[||s_θ(x̃, σ) - ∇_x̃ log p(x̃|x)||²]
    where x̃ = x + σ*z, z ~ N(0, I)
    """
    z = torch.randn_like(x)

    if sigma.dim() == 0:
        sigma = sigma.expand(x.shape[0])
    sigma_view = sigma.view(-1, 1, 1, 1)

    x_noisy = x + sigma_view * z
    target = -z / sigma_view
    score = score_net(x_noisy, labels)

    loss = 0.5 * ((score - target) ** 2).sum(dim=(1, 2, 3))
    loss = (loss * sigma ** 2).mean()

    return loss


def anneal_dsm_loss(
    score_net: nn.Module,
    x: torch.Tensor,
    sigmas: torch.Tensor,
    anneal_power: float = 2.0
) -> torch.Tensor:
    """
    Annealed Denoising Score Matching loss.

    Samples random noise level for each image and computes weighted DSM loss.
    Weight σ^anneal_power balances learning across noise levels.
    """
    labels = torch.randint(0, len(sigmas), (x.shape[0],), device=x.device)
    sigma = sigmas[labels].to(x.device)

    z = torch.randn_like(x)
    sigma_view = sigma.view(-1, 1, 1, 1)

    x_noisy = x + sigma_view * z
    target = -z / sigma_view
    score = score_net(x_noisy, labels)

    loss = 0.5 * ((score - target) ** 2).sum(dim=(1, 2, 3))
    loss = (loss * sigma ** anneal_power).mean()

    return loss
