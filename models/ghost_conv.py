"""
models/ghost_conv.py
====================
GhostConv backbone modules for YOLO-DarkWater.

Implementation strategy (frozen, per project spec):
  - Stem conv (layer 0, 3→64ch): kept as standard Conv — early low-level feature
    extraction (edges, textures, gradients) degrades with ghost approximation.
  - All other backbone + head convolutions: GhostConv / GhostC2f.

Uses Ultralytics' built-in GhostConv and GhostBottleneck as primitives —
they are already registered in parse_model's base_modules frozenset, ensuring
correct channel tracking without any additional patching.

GhostC2f is the only truly novel module: C2f with GhostBottleneck instead of
the standard Bottleneck. FLOPs savings come primarily from the bottleneck
replacement (each GhostBottleneck replaces two 3×3 standard convs).

References:
    Han et al., "GhostNet: More Features from Cheap Operations", CVPR 2020.
    https://arxiv.org/abs/1911.11907
"""

from __future__ import annotations

import torch
import torch.nn as nn

# Use Ultralytics' built-in Ghost primitives (already in parse_model known sets)
from ultralytics.nn.modules import GhostConv, GhostBottleneck
from ultralytics.nn.modules import Conv


__all__ = ["GhostConv", "GhostBottleneck", "GhostC2f"]


class GhostC2f(nn.Module):
    """
    Cross-Stage Partial with Ghost Bottlenecks (GhostC2f).

    Drop-in replacement for Ultralytics C2f — identical channel signature:
        GhostC2f(c1, c2, n=1, shortcut=False, g=1, e=0.5)

    Architecture (identical to C2f but with GhostBottleneck in .m):
        cv1: Conv(c1 → 2*c_)        split projection
        cv2: Conv((2+n)*c_ → c2)    output projection
        m:   [GhostBottleneck(c_, c_), ...]  × n blocks

    Why GhostBottleneck over standard Bottleneck:
        Each standard Bottleneck contains two Conv(k=3). GhostBottleneck replaces
        these with GhostConv operations, reducing FLOPs by ~40% per block while
        maintaining the cross-stage partial feature aggregation that makes C2f
        effective for multi-scale detection.

    IMPORTANT: This class is registered in models/__init__.py and injected into
    parse_model's build via post-processing (see __init__.py). The YAML parser
    first builds C2f for channel tracking, then __init__.py replaces those C2f
    instances with GhostC2f using the exact same (post-scaling) dimensions.

    Args:
        c1:       Input channels.
        c2:       Output channels.
        n:        Number of Ghost bottleneck blocks.
        shortcut: Enable residual connection inside bottleneck (c1==c2 check).
        g:        Groups (unused, kept for API parity with C2f).
        e:        Hidden channel expansion ratio.
    """

    def __init__(
        self,
        c1: int,
        c2: int,
        n: int = 1,
        shortcut: bool = False,
        g: int = 1,
        e: float = 0.5,
    ) -> None:
        super().__init__()
        self.c = int(c2 * e)  # hidden channels per stream

        # Split projection: c1 → 2 * hidden (split into two streams)
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)

        # Output projection: (2 + n) streams × hidden → c2
        self.cv2 = Conv((2 + n) * self.c, c2, 1)

        # Ghost bottleneck stack (replaces standard Bottleneck)
        # GhostBottleneck(c_, c_) uses stride=1 (standard in C2f)
        self.m = nn.ModuleList(
            GhostBottleneck(self.c, self.c) for _ in range(n)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass — identical flow to C2f:
        1. Project + split into 2 streams.
        2. Pass second stream sequentially through n Ghost bottlenecks.
        3. Concat all streams and project to output.
        """
        # Split cv1 output into two equal streams
        y = list(self.cv1(x).chunk(2, 1))
        # Apply Ghost bottlenecks sequentially on the second stream
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))
