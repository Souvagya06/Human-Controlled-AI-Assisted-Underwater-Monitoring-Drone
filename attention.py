import torch
import torch.nn as nn

class ChannelAttention(nn.Module):
    def __init__(self, c1=None, reduction=16):
        super().__init__()
        self.reduction = reduction
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.sigmoid = nn.Sigmoid()
        self.mlp = None

    def forward(self, x):
        if self.mlp is None:
            c1 = x.shape[1]
            reduced_c = max(1, c1 // self.reduction)
            self.mlp = nn.Sequential(
                nn.Conv2d(c1, reduced_c, 1, bias=False),
                nn.ReLU(inplace=True),
                nn.Conv2d(reduced_c, c1, 1, bias=False)
            ).to(x)
        avg_out = self.mlp(self.avg_pool(x))
        max_out = self.mlp(self.max_pool(x))
        return self.sigmoid(avg_out + max_out)

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super().__init__()
        assert kernel_size in (3, 7), "Kernel size must be 3 or 7"
        padding = 3 if kernel_size == 7 else 1
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x_cat = torch.cat([avg_out, max_out], dim=1)
        return self.sigmoid(self.conv(x_cat))

class CBAM(nn.Module):
    def __init__(self, c1=None, c2=None, reduction=16, kernel_size=7):
        super().__init__()
        self.channel_attn = ChannelAttention(c1, reduction=reduction)
        self.spatial_attn = SpatialAttention(kernel_size=kernel_size)

    def forward(self, x):
        x = x * self.channel_attn(x)
        x = x * self.spatial_attn(x)
        return x
