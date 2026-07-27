"""
Core Building Blocks
====================
Convolutional primitives, CSP-style feature blocks, spatial pooling, and the
position-sensitive attention block applied at P5.

These are shared by the backbone, neck, and head so that channel/activation
conventions stay consistent across the network.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def autopad(k, p=None, d=1):
    """Pad to preserve spatial size for odd kernels."""
    if d > 1:
        k = d * (k - 1) + 1
    if p is None:
        p = k // 2
    return p


class ConvBlock(nn.Module):
    """Conv + BN + SiLU."""

    def __init__(self, in_c, out_c, k=1, s=1, p=None, groups=1, d=1, act=True):
        super().__init__()
        self.conv = nn.Conv2d(in_c, out_c, k, s, autopad(k, p, d),
                              groups=groups, dilation=d, bias=False)
        self.bn = nn.BatchNorm2d(out_c)
        self.act = nn.SiLU(inplace=True) if act else nn.Identity()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

    def fuse_forward(self, x):
        """Forward without BN, for use after conv-bn fusion."""
        return self.act(self.conv(x))


class DepthwiseSeparableConv(nn.Module):
    """Depthwise 3x3 followed by pointwise 1x1."""

    def __init__(self, in_c, out_c, k=3, s=1, p=None):
        super().__init__()
        self.depthwise = ConvBlock(in_c, in_c, k, s, p, groups=in_c)
        self.pointwise = ConvBlock(in_c, out_c, k=1)

    def forward(self, x):
        return self.pointwise(self.depthwise(x))


class Bottleneck(nn.Module):
    """Standard residual bottleneck used inside CSP blocks."""

    def __init__(self, in_c, out_c, shortcut=True, expansion=0.5, k=(3, 3)):
        super().__init__()
        hidden = int(out_c * expansion)
        self.cv1 = ConvBlock(in_c, hidden, k=k[0])
        self.cv2 = ConvBlock(hidden, out_c, k=k[1])
        self.add = shortcut and in_c == out_c

    def forward(self, x):
        out = self.cv2(self.cv1(x))
        return x + out if self.add else out


class C2f(nn.Module):
    """CSP block with two convolutions and dense bottleneck routing.

    Every bottleneck output is concatenated into the fusion, which gives more
    gradient paths than the classic CSP split at similar cost.
    """

    def __init__(self, in_c, out_c, n=1, shortcut=True, expansion=0.5):
        super().__init__()
        self.hidden = int(out_c * expansion)
        self.cv1 = ConvBlock(in_c, 2 * self.hidden, k=1)
        self.cv2 = ConvBlock((2 + n) * self.hidden, out_c, k=1)
        self.blocks = nn.ModuleList(
            Bottleneck(self.hidden, self.hidden, shortcut, expansion=1.0)
            for _ in range(n)
        )

    def forward(self, x):
        y = list(self.cv1(x).chunk(2, dim=1))
        for block in self.blocks:
            y.append(block(y[-1]))
        return self.cv2(torch.cat(y, dim=1))


class SPPF(nn.Module):
    """Spatial pyramid pooling, fast variant.

    Three sequential 5x5 max-pools approximate 5/9/13 pooling at a fraction of
    the cost, widening the receptive field before the neck.
    """

    def __init__(self, in_c, out_c, k=5):
        super().__init__()
        hidden = in_c // 2
        self.cv1 = ConvBlock(in_c, hidden, k=1)
        self.cv2 = ConvBlock(hidden * 4, out_c, k=1)
        self.pool = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)

    def forward(self, x):
        x = self.cv1(x)
        y1 = self.pool(x)
        y2 = self.pool(y1)
        y3 = self.pool(y2)
        return self.cv2(torch.cat([x, y1, y2, y3], dim=1))


class Attention(nn.Module):
    """Multi-head self-attention over a flattened feature map.

    Includes a depthwise positional-encoding convolution on the value tensor,
    which supplies spatial inductive bias that plain attention lacks.
    """

    def __init__(self, dim, num_heads=4, attn_ratio=0.5):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.key_dim = int(self.head_dim * attn_ratio)
        self.scale = self.key_dim ** -0.5

        qk_dim = self.key_dim * num_heads
        self.qkv = ConvBlock(dim, qk_dim * 2 + dim, k=1, act=False)
        self.proj = ConvBlock(dim, dim, k=1, act=False)
        self.pe = ConvBlock(dim, dim, k=3, groups=dim, act=False)

    def forward(self, x):
        b, c, h, w = x.shape
        n = h * w
        qkv = self.qkv(x)
        qkv = qkv.view(b, self.num_heads, self.key_dim * 2 + self.head_dim, n)
        q, k, v = qkv.split([self.key_dim, self.key_dim, self.head_dim], dim=2)

        attn = (q.transpose(-2, -1) @ k) * self.scale
        attn = attn.softmax(dim=-1)

        out = (v @ attn.transpose(-2, -1)).view(b, c, h, w)
        out = out + self.pe(v.reshape(b, c, h, w))
        return self.proj(out)


class PSABlock(nn.Module):
    """Position-sensitive attention block: attention + feed-forward, both residual."""

    def __init__(self, dim, num_heads=4, expansion=2.0):
        super().__init__()
        self.attn = Attention(dim, num_heads=num_heads)
        hidden = int(dim * expansion)
        self.ffn = nn.Sequential(
            ConvBlock(dim, hidden, k=1),
            ConvBlock(hidden, dim, k=1, act=False),
        )

    def forward(self, x):
        x = x + self.attn(x)
        return x + self.ffn(x)


class C2PSA(nn.Module):
    """CSP-style wrapper around stacked PSA blocks.

    Half the channels pass through attention while half skip it, so the block
    adds global context without discarding local convolutional features. This
    is applied at P5 only, where the token count is small enough to be cheap.
    """

    def __init__(self, in_c, out_c, n=1, num_heads=4):
        super().__init__()
        assert in_c == out_c, "C2PSA expects matching input/output channels"
        self.hidden = in_c // 2
        self.cv1 = ConvBlock(in_c, 2 * self.hidden, k=1)
        self.cv2 = ConvBlock(2 * self.hidden, out_c, k=1)
        self.blocks = nn.Sequential(
            *(PSABlock(self.hidden, num_heads=max(1, num_heads // 2)) for _ in range(n))
        )

    def forward(self, x):
        a, b = self.cv1(x).chunk(2, dim=1)
        return self.cv2(torch.cat([a, self.blocks(b)], dim=1))


class DFL(nn.Module):
    """Distribution Focal Loss integral.

    Converts the predicted per-side distribution over `reg_max+1` bins into a
    scalar distance by taking its expectation. Implemented as a frozen 1x1
    convolution holding the bin indices as weights.
    """

    def __init__(self, reg_max=16):
        super().__init__()
        self.reg_max = reg_max
        self.conv = nn.Conv2d(reg_max, 1, kernel_size=1, bias=False)
        weight = torch.arange(reg_max, dtype=torch.float32).view(1, reg_max, 1, 1)
        self.conv.weight.data.copy_(weight)
        self.conv.weight.requires_grad_(False)

    def forward(self, x):
        """x: [B, 4*reg_max, A] -> [B, 4, A]"""
        b, _, a = x.shape
        x = x.view(b, 4, self.reg_max, a).transpose(1, 2)
        x = x.softmax(dim=1)
        return self.conv(x).view(b, 4, a)
