"""UNet generator architecture.

Adapted from:
    C.-W. Huang, J. H. Lim, and A. Courville.
    A variational perspective on diffusion-based generative models and score matching.
    NeurIPS, 2021.

Original code: https://github.com/CW-Huang/sdeflow-light
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.init import _calculate_fan_in_and_fan_out


# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------

class Swish(nn.Module):
    def forward(self, x):
        return torch.sigmoid(x) * x


def group_norm(out_ch):
    return nn.GroupNorm(num_groups=32, num_channels=out_ch, eps=1e-6, affine=True)


def _calculate_correct_fan(tensor, mode):
    mode = mode.lower()
    valid_modes = ['fan_in', 'fan_out', 'fan_avg']
    if mode not in valid_modes:
        raise ValueError(f"Mode {mode} not supported, use one of {valid_modes}")
    fan_in, fan_out = _calculate_fan_in_and_fan_out(tensor)
    return fan_in if mode == 'fan_in' else fan_out


def kaiming_uniform_(tensor, gain=1.0, mode='fan_in'):
    fan = _calculate_correct_fan(tensor, mode)
    var = gain / max(1.0, fan)
    bound = math.sqrt(3.0 * var)
    with torch.no_grad():
        return tensor.uniform_(-bound, bound)


def variance_scaling_init_(tensor, scale):
    return kaiming_uniform_(tensor, gain=1e-10 if scale == 0 else scale, mode='fan_avg')


def dense(in_channels, out_channels, init_scale=1.0):
    lin = nn.Linear(in_channels, out_channels)
    variance_scaling_init_(lin.weight, scale=init_scale)
    nn.init.zeros_(lin.bias)
    return lin


def conv2d(in_planes, out_planes, kernel_size=(3, 3), stride=1, dilation=1,
           padding=1, bias=True, padding_mode='zeros', init_scale=1.0):
    conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size,
                     stride=stride, padding=padding, dilation=dilation,
                     bias=bias, padding_mode=padding_mode)
    variance_scaling_init_(conv.weight, scale=init_scale)
    if bias:
        nn.init.zeros_(conv.bias)
    return conv


def upsample(in_ch, with_conv):
    up = nn.Sequential()
    up.add_module('up_nn', nn.Upsample(scale_factor=2, mode='nearest'))
    if with_conv:
        up.add_module('up_conv', conv2d(in_ch, in_ch, kernel_size=(3, 3), stride=1))
    return up


def downsample(in_ch, with_conv):
    if with_conv:
        return conv2d(in_ch, in_ch, kernel_size=(3, 3), stride=2)
    return nn.AvgPool2d(2, 2)


# ---------------------------------------------------------------------------
# Residual block and self-attention
# ---------------------------------------------------------------------------

class ResidualBlock(nn.Module):
    def __init__(self, in_ch, out_ch=None, conv_shortcut=False, dropout=0.0,
                 normalize=group_norm, act=Swish()):
        super().__init__()
        self.in_ch = in_ch
        self.out_ch = out_ch if out_ch is not None else in_ch
        self.act = act

        self.norm1 = normalize(in_ch) if normalize is not None else nn.Identity()
        self.conv1 = conv2d(in_ch, self.out_ch)
        self.norm2 = normalize(self.out_ch) if normalize is not None else nn.Identity()
        self.dropout = nn.Dropout2d(p=dropout) if dropout > 0.0 else nn.Identity()
        self.conv2 = conv2d(self.out_ch, self.out_ch, init_scale=0.0)

        if in_ch != self.out_ch:
            if conv_shortcut:
                self.shortcut = conv2d(in_ch, self.out_ch)
            else:
                self.shortcut = conv2d(in_ch, self.out_ch, kernel_size=(1, 1), padding=0)
        else:
            self.shortcut = nn.Identity()

    def forward(self, x):
        h = self.act(self.norm1(x))
        h = self.conv1(h)
        h = self.act(self.norm2(h))
        h = self.dropout(h)
        h = self.conv2(h)
        return self.shortcut(x) + h


class SelfAttention(nn.Module):
    def __init__(self, in_channels, normalize=group_norm):
        super().__init__()
        self.in_channels = in_channels
        self.attn_q = conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0)
        self.attn_k = conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0)
        self.attn_v = conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0)
        self.proj_out = conv2d(in_channels, in_channels, kernel_size=1, stride=1,
                               padding=0, init_scale=0.0)
        self.softmax = nn.Softmax(dim=-1)
        self.norm = normalize(in_channels) if normalize is not None else nn.Identity()

    def forward(self, x):
        _, C, H, W = x.size()
        h = self.norm(x)
        q = self.attn_q(h).view(-1, C, H * W)
        k = self.attn_k(h).view(-1, C, H * W)
        v = self.attn_v(h).view(-1, C, H * W)

        attn = torch.bmm(q.permute(0, 2, 1), k) * (int(C) ** (-0.5))
        attn = self.softmax(attn)

        h = torch.bmm(v, attn.permute(0, 2, 1))
        h = h.view(-1, C, H, W)
        h = self.proj_out(h)
        return x + h


# ---------------------------------------------------------------------------
# UNet
# ---------------------------------------------------------------------------

class UNet(nn.Module):
    """UNet generator that maps noisy images to images (same spatial size)."""

    def __init__(self, input_channels, input_height, ch, output_channels=None,
                 ch_mult=(1, 2, 4, 8), num_res_blocks=2, attn_resolutions=(16,),
                 dropout=0.0, resamp_with_conv=True, act=Swish(),
                 normalize=group_norm):
        super().__init__()
        self.input_channels = input_channels
        self.input_height = input_height
        self.ch = ch
        self.output_channels = input_channels if output_channels is None else output_channels
        self.ch_mult = ch_mult
        self.num_res_blocks = num_res_blocks
        self.attn_resolutions = attn_resolutions
        self.num_resolutions = len(ch_mult)
        self.act = act

        in_ht = input_height
        assert in_ht % 2 ** (self.num_resolutions - 1) == 0, \
            "input_height must be divisible by 2^(num_resolutions-1)"

        # Downsampling
        self.begin_conv = conv2d(input_channels, ch)
        unet_chs = [ch]
        in_ch = ch
        down_modules = []
        for i_level in range(self.num_resolutions):
            block_modules = {}
            out_ch = ch * ch_mult[i_level]
            for i_block in range(num_res_blocks):
                block_modules[f'{i_level}a_{i_block}a_block'] = ResidualBlock(
                    in_ch=in_ch, out_ch=out_ch, dropout=dropout,
                    act=act, normalize=normalize)
                if in_ht in attn_resolutions:
                    block_modules[f'{i_level}a_{i_block}b_attn'] = SelfAttention(
                        out_ch, normalize=normalize)
                unet_chs.append(out_ch)
                in_ch = out_ch
            if i_level != self.num_resolutions - 1:
                block_modules[f'{i_level}b_downsample'] = downsample(
                    out_ch, with_conv=resamp_with_conv)
                in_ht //= 2
                unet_chs.append(out_ch)
            down_modules.append(nn.ModuleDict(block_modules))
        self.down_modules = nn.ModuleList(down_modules)

        # Middle
        self.mid_modules = nn.ModuleList([
            ResidualBlock(in_ch, out_ch=in_ch, dropout=dropout, act=act, normalize=normalize),
            SelfAttention(in_ch, normalize=normalize),
            ResidualBlock(in_ch, out_ch=in_ch, dropout=dropout, act=act, normalize=normalize),
        ])

        # Upsampling
        up_modules = []
        for i_level in reversed(range(self.num_resolutions)):
            block_modules = {}
            out_ch = ch * ch_mult[i_level]
            for i_block in range(num_res_blocks + 1):
                block_modules[f'{i_level}a_{i_block}a_block'] = ResidualBlock(
                    in_ch=in_ch + unet_chs.pop(), out_ch=out_ch, dropout=dropout,
                    act=act, normalize=normalize)
                if in_ht in attn_resolutions:
                    block_modules[f'{i_level}a_{i_block}b_attn'] = SelfAttention(
                        out_ch, normalize=normalize)
                in_ch = out_ch
            if i_level != 0:
                block_modules[f'{i_level}b_upsample'] = upsample(
                    out_ch, with_conv=resamp_with_conv)
                in_ht *= 2
            up_modules.append(nn.ModuleDict(block_modules))
        self.up_modules = nn.ModuleList(up_modules)
        assert not unet_chs, "UNet channel bookkeeping mismatch"

        # Output
        self.end_conv = nn.Sequential(
            normalize(in_ch), self.act,
            conv2d(in_ch, self.output_channels, init_scale=0.0))

    def _compute_cond_module(self, module, x):
        for m in module:
            x = m(x)
        return x

    def forward(self, x):
        # Downsampling
        hs = [self.begin_conv(x)]
        for i_level in range(self.num_resolutions):
            block_modules = self.down_modules[i_level]
            for i_block in range(self.num_res_blocks):
                h = block_modules[f'{i_level}a_{i_block}a_block'](hs[-1])
                if h.size(2) in self.attn_resolutions:
                    h = block_modules[f'{i_level}a_{i_block}b_attn'](h)
                hs.append(h)
            if i_level != self.num_resolutions - 1:
                hs.append(block_modules[f'{i_level}b_downsample'](hs[-1]))

        # Middle
        h = self._compute_cond_module(self.mid_modules, hs[-1])

        # Upsampling
        for i_idx, i_level in enumerate(reversed(range(self.num_resolutions))):
            block_modules = self.up_modules[i_idx]
            for i_block in range(self.num_res_blocks + 1):
                h = block_modules[f'{i_level}a_{i_block}a_block'](
                    torch.cat([h, hs.pop()], dim=1))
                if h.size(2) in self.attn_resolutions:
                    h = block_modules[f'{i_level}a_{i_block}b_attn'](h)
            if i_level != 0:
                h = block_modules[f'{i_level}b_upsample'](h)
        assert not hs

        return self.end_conv(h)
