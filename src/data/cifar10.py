"""
CIFAR-10 Data Loading
=====================
Provides data loading for CIFAR-10.
Labels are not used - we focus on learning p(x).
"""

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


def get_dataloader(
    batch_size: int = 128,
    train: bool = True,
    num_workers: int = 2,
    data_dir: str = './data'
) -> DataLoader:
    """
    Get CIFAR-10 dataloader normalized to [-1, 1].
    """
    transform_list = []

    if train:
        transform_list.append(transforms.RandomHorizontalFlip())

    transform_list.extend([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    transform = transforms.Compose(transform_list)

    dataset = datasets.CIFAR10(
        root=data_dir,
        train=train,
        download=True,
        transform=transform
    )

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=train,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=train
    )


def denormalize(x: torch.Tensor) -> torch.Tensor:
    """Denormalize from [-1, 1] to [0, 1]."""
    return (x + 1) / 2
