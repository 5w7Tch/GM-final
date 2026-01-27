"""
Visualization Utilities
=======================
For displaying and saving generated samples.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from torchvision.utils import make_grid, save_image


def denormalize(x: torch.Tensor) -> torch.Tensor:
    """Denormalize from [-1, 1] to [0, 1]."""
    return (x + 1) / 2


def show_samples(
    samples: torch.Tensor,
    nrow: int = 8,
    title: str = None,
    figsize: tuple = (10, 10)
):
    """Display a grid of samples."""
    samples = denormalize(samples.cpu())
    grid = make_grid(samples, nrow=nrow, padding=2)
    grid = grid.permute(1, 2, 0).numpy()
    grid = np.clip(grid, 0, 1)

    plt.figure(figsize=figsize)
    plt.imshow(grid)
    plt.axis('off')
    if title:
        plt.title(title)
    plt.tight_layout()
    plt.show()


def save_samples(
    samples: torch.Tensor,
    path: str,
    nrow: int = 8
):
    """Save samples to file."""
    samples = denormalize(samples)
    save_image(samples, path, nrow=nrow)
