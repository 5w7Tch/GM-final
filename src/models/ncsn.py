"""
Noise Conditional Score Network (NCSN)
======================================
Based on: "Generative Modeling by Estimating Gradients of the Data Distribution"
Song & Ermon, NeurIPS 2019

The model learns the score function ∇_x log p(x) at multiple noise levels.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional, Tuple


class ConditionalInstanceNorm2d(nn.Module):
    """Conditional Instance Normalization based on noise level index."""

    def __init__(self, num_features: int, num_classes: int):
        super().__init__()
        self.num_features = num_features
        self.instance_norm = nn.InstanceNorm2d(num_features, affine=False)
        self.gamma = nn.Embedding(num_classes, num_features)
        self.beta = nn.Embedding(num_classes, num_features)
        self.gamma.weight.data.fill_(1.0)
        self.beta.weight.data.zero_()

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        out = self.instance_norm(x)
        gamma = self.gamma(y).view(-1, self.num_features, 1, 1)
        beta = self.beta(y).view(-1, self.num_features, 1, 1)
        return gamma * out + beta


class ConditionalResBlock(nn.Module):
    """Residual block with conditional instance normalization."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        num_classes: int,
        resample: Optional[str] = None
    ):
        super().__init__()
        self.resample = resample
        self.activation = nn.ELU()

        self.norm1 = ConditionalInstanceNorm2d(in_channels, num_classes)
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.norm2 = ConditionalInstanceNorm2d(out_channels, num_classes)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)

        if in_channels != out_channels or resample is not None:
            self.shortcut = nn.Conv2d(in_channels, out_channels, 1)
        else:
            self.shortcut = nn.Identity()

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        shortcut = x
        out = self.activation(self.norm1(x, y))

        if self.resample == 'down':
            out = F.avg_pool2d(out, 2)
            shortcut = F.avg_pool2d(shortcut, 2)
        elif self.resample == 'up':
            out = F.interpolate(out, scale_factor=2, mode='nearest')
            shortcut = F.interpolate(shortcut, scale_factor=2, mode='nearest')

        out = self.conv1(out)
        out = self.activation(self.norm2(out, y))
        out = self.conv2(out)

        return out + self.shortcut(shortcut)


class RefineBlock(nn.Module):
    """RefineNet-style block for multi-scale feature fusion."""

    def __init__(self, in_channels: list, out_channels: int, num_classes: int):
        super().__init__()
        self.activation = nn.ELU()

        self.adapt_convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(ch, out_channels, 3, padding=1),
                nn.Conv2d(out_channels, out_channels, 3, padding=1)
            ) for ch in in_channels
        ])

        self.norm = ConditionalInstanceNorm2d(out_channels, num_classes)
        self.conv1 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)

    def forward(
        self,
        inputs: list,
        y: torch.Tensor,
        output_size: Tuple[int, int]
    ) -> torch.Tensor:
        adapted = []
        for x, conv in zip(inputs, self.adapt_convs):
            h = conv(x)
            if h.shape[2:] != output_size:
                h = F.interpolate(h, size=output_size, mode='bilinear', align_corners=True)
            adapted.append(h)

        out = sum(adapted)
        out = self.activation(out)
        out = self.activation(self.norm(out, y))
        out = self.conv1(out)
        out = self.activation(out)
        out = self.conv2(out)
        return out


class NCSN(nn.Module):
    """
    Noise Conditional Score Network.

    A RefineNet-based architecture that estimates the score function
    ∇_x log p_σ(x) for multiple noise levels σ.
    """

    def __init__(
        self,
        num_classes: int = 10,
        num_features: int = 128,
        image_channels: int = 3
    ):
        super().__init__()
        self.num_classes = num_classes
        self.activation = nn.ELU()
        nf = num_features

        # Input
        self.input_conv = nn.Conv2d(image_channels, nf, 3, padding=1)

        # Encoder
        self.res1 = ConditionalResBlock(nf, nf, num_classes)
        self.res2 = ConditionalResBlock(nf, nf, num_classes)
        self.down1 = ConditionalResBlock(nf, nf * 2, num_classes, resample='down')

        self.res3 = ConditionalResBlock(nf * 2, nf * 2, num_classes)
        self.res4 = ConditionalResBlock(nf * 2, nf * 2, num_classes)
        self.down2 = ConditionalResBlock(nf * 2, nf * 2, num_classes, resample='down')

        self.res5 = ConditionalResBlock(nf * 2, nf * 2, num_classes)
        self.res6 = ConditionalResBlock(nf * 2, nf * 2, num_classes)
        self.down3 = ConditionalResBlock(nf * 2, nf * 4, num_classes, resample='down')

        self.res7 = ConditionalResBlock(nf * 4, nf * 4, num_classes)
        self.res8 = ConditionalResBlock(nf * 4, nf * 4, num_classes)

        # RefineNet decoder
        self.refine4 = RefineBlock([nf * 4], nf * 4, num_classes)
        self.refine3 = RefineBlock([nf * 4, nf * 2], nf * 2, num_classes)
        self.refine2 = RefineBlock([nf * 2, nf * 2], nf * 2, num_classes)
        self.refine1 = RefineBlock([nf * 2, nf], nf, num_classes)

        # Output
        self.output_norm = ConditionalInstanceNorm2d(nf, num_classes)
        self.output_conv = nn.Conv2d(nf, image_channels, 3, padding=1)

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        h = self.input_conv(x)

        # Level 1 (32x32)
        h1 = self.res2(self.res1(h, y), y)
        h = self.down1(h1, y)

        # Level 2 (16x16)
        h2 = self.res4(self.res3(h, y), y)
        h = self.down2(h2, y)

        # Level 3 (8x8)
        h3 = self.res6(self.res5(h, y), y)
        h = self.down3(h3, y)

        # Level 4 (4x4)
        h4 = self.res8(self.res7(h, y), y)

        # RefineNet upsampling
        h = self.refine4([h4], y, (4, 4))
        h = self.refine3([h, h3], y, (8, 8))
        h = self.refine2([h, h2], y, (16, 16))
        h = self.refine1([h, h1], y, (32, 32))

        # Output
        h = self.activation(self.output_norm(h, y))
        return self.output_conv(h)


class NCSNv2(nn.Module):
    """
    Improved NCSN with U-Net style skip connections.
    Based on: "Improved Techniques for Training Score-Based Generative Models"
    """

    def __init__(
        self,
        num_classes: int = 10,
        num_features: int = 128,
        image_channels: int = 3
    ):
        super().__init__()
        self.num_classes = num_classes
        self.activation = nn.ELU()
        nf = num_features

        self.begin_conv = nn.Conv2d(image_channels, nf, 3, padding=1)

        # Encoder
        self.enc1 = nn.ModuleList([
            ConditionalResBlock(nf, nf, num_classes),
            ConditionalResBlock(nf, nf, num_classes),
        ])
        self.enc2 = nn.ModuleList([
            ConditionalResBlock(nf, nf * 2, num_classes, resample='down'),
            ConditionalResBlock(nf * 2, nf * 2, num_classes),
        ])
        self.enc3 = nn.ModuleList([
            ConditionalResBlock(nf * 2, nf * 2, num_classes, resample='down'),
            ConditionalResBlock(nf * 2, nf * 2, num_classes),
        ])
        self.enc4 = nn.ModuleList([
            ConditionalResBlock(nf * 2, nf * 4, num_classes, resample='down'),
            ConditionalResBlock(nf * 4, nf * 4, num_classes),
        ])

        # Decoder
        self.dec4 = nn.ModuleList([
            ConditionalResBlock(nf * 4, nf * 4, num_classes),
        ])
        self.dec3 = nn.ModuleList([
            ConditionalResBlock(nf * 4 + nf * 2, nf * 2, num_classes),
            ConditionalResBlock(nf * 2, nf * 2, num_classes, resample='up'),
        ])
        self.dec2 = nn.ModuleList([
            ConditionalResBlock(nf * 2 + nf * 2, nf * 2, num_classes),
            ConditionalResBlock(nf * 2, nf * 2, num_classes, resample='up'),
        ])
        self.dec1 = nn.ModuleList([
            ConditionalResBlock(nf * 2 + nf, nf, num_classes),
            ConditionalResBlock(nf, nf, num_classes, resample='up'),
        ])

        self.final_norm = ConditionalInstanceNorm2d(nf, num_classes)
        self.final_conv = nn.Conv2d(nf, image_channels, 3, padding=1)

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        h = self.begin_conv(x)

        # Encoder
        enc_outputs = []
        for block in self.enc1:
            h = block(h, y)
        enc_outputs.append(h)

        for block in self.enc2:
            h = block(h, y)
        enc_outputs.append(h)

        for block in self.enc3:
            h = block(h, y)
        enc_outputs.append(h)

        for block in self.enc4:
            h = block(h, y)

        # Decoder
        for block in self.dec4:
            h = block(h, y)

        h = torch.cat([h, enc_outputs[2]], dim=1)
        for block in self.dec3:
            h = block(h, y)

        h = torch.cat([h, enc_outputs[1]], dim=1)
        for block in self.dec2:
            h = block(h, y)

        h = torch.cat([h, enc_outputs[0]], dim=1)
        for block in self.dec1:
            h = block(h, y)

        h = self.activation(self.final_norm(h, y))
        return self.final_conv(h)


def get_sigmas(
    sigma_begin: float = 1.0,
    sigma_end: float = 0.01,
    num_classes: int = 10
) -> torch.Tensor:
    """Get geometric sequence of noise levels (σ values)."""
    return torch.tensor(
        np.geomspace(sigma_begin, sigma_end, num_classes),
        dtype=torch.float32
    )
