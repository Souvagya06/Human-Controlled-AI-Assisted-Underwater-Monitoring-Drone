"""
models/attention.py
====================
CBAM (Convolutional Block Attention Module) for YOLO-DarkWater.

Replaces the previous placeholder with a complete, production-quality
implementation that is fully compatible with the Ultralytics YAML parser.

Architecture:
    CBAM(x) = SpatialAttention( ChannelAttention(x) · x ) · ChannelAttention(x) · x

    1. Channel Attention:
       Uses both AvgPool and MaxPool → shared MLP → sigmoid gate.
       Suppresses noisy/irrelevant channels (e.g., red-channel attenuation
       artefacts in turbid underwater imagery).

    2. Spatial Attention:
       Concatenates channel-wise avg and max → 7×7 conv → sigmoid gate.
       Focuses on spatially discriminative debris regions.

Placement in YOLO-DarkWater:
    After P3 feature map (256 channels). This provides the best tradeoff:
    - P3 retains high-resolution spatial detail → spatial attention is effective
    - 256ch is manageable overhead; P4/P5 would double cost with less benefit
      on this small dataset

References:
    Woo et al., "CBAM: Convolutional Block Attention Module", ECCV 2018.
    https://arxiv.org/abs/1807.06521

Versioning:
    model.version = "1.0"  (set in DarkWaterTrainer after model construction)
"""

from __future__ import annotations

import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# Channel Attention
# ---------------------------------------------------------------------------

class ChannelAttention(nn.Module):
    """
    Channel Attention Module.

    Applies both average and max global pooling, passes through a shared
    two-layer MLP (implemented as 1×1 convs for spatial flexibility),
    and combines via element-wise addition + sigmoid.

    Args:
        channels:   Number of input/output channels.
        reduction:  Channel reduction ratio for the MLP bottleneck.
                    Default 16 (standard from paper); use 8 for small channel counts.
    """

    def __init__(self, channels: int, reduction: int = 16) -> None:
        super().__init__()
        # Clamp reduction to avoid degenerate bottleneck (e.g., 8ch / 16 = 0)
        reduced = max(1, channels // reduction)

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        # Shared MLP — implemented as 1×1 convolutions (equivalent to Linear)
        self.shared_mlp = nn.Sequential(
            nn.Conv2d(channels, reduced, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(reduced, channels, kernel_size=1, bias=False),
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg_out = self.shared_mlp(self.avg_pool(x))
        max_out = self.shared_mlp(self.max_pool(x))
        scale = self.sigmoid(avg_out + max_out)          # (B, C, 1, 1)
        return x * scale


# ---------------------------------------------------------------------------
# Spatial Attention
# ---------------------------------------------------------------------------

class SpatialAttention(nn.Module):
    """
    Spatial Attention Module.

    Applies channel-wise AvgPool and MaxPool, concatenates the results
    along the channel dim (2 channels total), then applies a 7×7 conv
    to produce a spatial attention map.

    Args:
        kernel_size: Convolution kernel for the spatial map. 7 (default, per paper)
                     produces larger receptive field; 3 is lighter.
    """

    def __init__(self, kernel_size: int = 7) -> None:
        super().__init__()
        padding = kernel_size // 2
        self.conv = nn.Conv2d(
            2, 1, kernel_size=kernel_size, padding=padding, bias=False
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg_out = torch.mean(x, dim=1, keepdim=True)       # (B, 1, H, W)
        max_out, _ = torch.max(x, dim=1, keepdim=True)     # (B, 1, H, W)
        concat = torch.cat([avg_out, max_out], dim=1)      # (B, 2, H, W)
        scale = self.sigmoid(self.conv(concat))             # (B, 1, H, W)
        return x * scale


# ---------------------------------------------------------------------------
# CBAM — full module
# ---------------------------------------------------------------------------

class CBAM(nn.Module):
    """
    Convolutional Block Attention Module (CBAM).

    Sequentially applies Channel Attention then Spatial Attention.
    Designed as a drop-in feature refinement block.

    Ultralytics YAML compatibility:
        The constructor accepts the Ultralytics YAML argument convention:
            CBAM(c1, c2, reduction, kernel_size)
        where c2 is unused (output channels == input channels) but included
        for API parity with other Ultralytics modules.

    Args:
        c1:          Number of input channels (= output channels).
        c2:          Ignored — included for Ultralytics YAML parser compatibility.
        reduction:   Channel attention reduction ratio.
        kernel_size: Spatial attention kernel size.
    """

    VERSION: str = "1.0"
    MODEL_NAME: str = "YOLO-DarkWater"

    def __init__(
        self,
        c1: int,
        c2: int | None = None,        # unused; kept for YAML parser compatibility
        reduction: int = 16,
        kernel_size: int = 7,
    ) -> None:
        super().__init__()
        self.channel_attention = ChannelAttention(c1, reduction)
        self.spatial_attention = SpatialAttention(kernel_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.channel_attention(x)
        x = self.spatial_attention(x)
        return x