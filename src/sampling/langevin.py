"""
Langevin Dynamics Sampling
==========================
For generating samples from NCSN using the learned score function.

x_{t+1} = x_t + ε * ∇_x log p(x_t) + √(2ε) * z_t
"""

import torch
import torch.nn as nn
from typing import Optional, Callable
from tqdm import tqdm


@torch.no_grad()
def langevin_dynamics(
    score_net: nn.Module,
    x_init: torch.Tensor,
    sigma: float,
    label: int,
    n_steps: int = 100,
    step_size: float = None,
    clamp: bool = True
) -> torch.Tensor:
    """
    Langevin Monte Carlo at a single noise level.
    """
    device = x_init.device
    x = x_init.clone()

    if step_size is None:
        step_size = (sigma ** 2) * 0.1

    labels = torch.full((x.shape[0],), label, dtype=torch.long, device=device)

    for _ in range(n_steps):
        score = score_net(x, labels)
        noise = torch.randn_like(x)
        x = x + step_size * score + (2 * step_size) ** 0.5 * noise

        if clamp:
            x = x.clamp(-1, 1)

    return x


@torch.no_grad()
def anneal_langevin_dynamics(
    score_net: nn.Module,
    x_init: torch.Tensor,
    sigmas: torch.Tensor,
    n_steps_each: int = 100,
    step_lr: float = 2e-5,
    denoise: bool = True,
    verbose: bool = True,
    callback: Optional[Callable] = None
) -> torch.Tensor:
    """
    Annealed Langevin Dynamics.

    Performs Langevin dynamics from high to low noise levels,
    using result from each level as initialization for the next.
    """
    device = x_init.device
    x = x_init.clone()
    sigmas = sigmas.to(device)

    iterator = enumerate(sigmas)
    if verbose:
        iterator = tqdm(list(iterator), desc='Annealed Langevin')

    for i, sigma in iterator:
        labels = torch.full((x.shape[0],), i, dtype=torch.long, device=device)
        step_size = step_lr * (sigma / sigmas[-1]) ** 2

        for _ in range(n_steps_each):
            score = score_net(x, labels)
            noise = torch.randn_like(x)
            x = x + step_size * score + (2 * step_size) ** 0.5 * noise
            x = x.clamp(-1, 1)

        if callback is not None:
            callback(x, i, sigma.item())

    # Final denoising
    if denoise:
        labels = torch.full((x.shape[0],), len(sigmas) - 1, dtype=torch.long, device=device)
        x = x + sigmas[-1] ** 2 * score_net(x, labels)
        x = x.clamp(-1, 1)

    return x


@torch.no_grad()
def generate_samples(
    score_net: nn.Module,
    sigmas: torch.Tensor,
    n_samples: int = 64,
    image_size: int = 32,
    channels: int = 3,
    n_steps_each: int = 100,
    step_lr: float = 2e-5,
    device: str = 'cuda',
    verbose: bool = True
) -> torch.Tensor:
    """
    Generate samples using annealed Langevin dynamics.
    """
    score_net.eval()

    # Initialize from uniform noise [-1, 1]
    x_init = torch.rand(n_samples, channels, image_size, image_size, device=device)
    x_init = x_init * 2 - 1

    samples = anneal_langevin_dynamics(
        score_net=score_net,
        x_init=x_init,
        sigmas=sigmas,
        n_steps_each=n_steps_each,
        step_lr=step_lr,
        denoise=True,
        verbose=verbose
    )

    return samples
