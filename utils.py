"""Utility functions: EMA, network helpers, Jacobian norms, and misc."""

import os
import copy
import math
import warnings

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.nn.init import _calculate_fan_in_and_fan_out


# ---------------------------------------------------------------------------
# Directory / tensor helpers
# ---------------------------------------------------------------------------

def ensure_dir(path: str):
    """Create directory (and parents) if it doesn't exist."""
    os.makedirs(path, exist_ok=True)


def to_0_1(imgs: torch.Tensor, images_in_0_1: bool) -> torch.Tensor:
    """Convert image tensor to [0, 1] range for saving."""
    if images_in_0_1:
        return imgs.clamp(0, 1)
    return ((imgs + 1.0) / 2.0).clamp(0, 1)


def save_2d_scatter(real_samples, fake_samples, path, title=""):
    """Save a 2D scatter plot of real vs generated samples."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(1, 1, figsize=(6, 6))
    real = real_samples.detach().cpu().numpy()
    fake = fake_samples.detach().cpu().numpy()
    ax.scatter(real[:, 0], real[:, 1], s=4, alpha=0.4, label="Real", c="tab:blue")
    ax.scatter(fake[:, 0], fake[:, 1], s=4, alpha=0.4, label="Generated", c="tab:red")
    ax.legend(fontsize=9)
    ax.set_title(title)
    ax.set_aspect("equal")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Network building blocks
# ---------------------------------------------------------------------------

def conv3x3(in_ch, out_ch, bias=True):
    return nn.Conv2d(in_ch, out_ch, 3, 1, 1, bias=bias)


def conv1x1(in_ch, out_ch, bias=True):
    return nn.Conv2d(in_ch, out_ch, 1, 1, 0, bias=bias)


def safe_gn_groups(channels: int) -> int:
    """Return the largest valid group count for GroupNorm."""
    for g in (32, 16, 8, 4, 2, 1):
        if channels % g == 0:
            return g
    return 1


def center_pad_to(x: torch.Tensor, target_hw: tuple) -> torch.Tensor:
    """Centre-pad (or crop) spatial dims to target (H, W)."""
    B, C, H, W = x.shape
    th, tw = target_hw
    # Crop if bigger
    if H > th:
        top = (H - th) // 2
        x = x[:, :, top:top + th, :]
    if W > tw:
        left = (W - tw) // 2
        x = x[:, :, :, left:left + tw]
    # Pad if smaller
    B, C, H, W = x.shape
    pad_h, pad_w = th - H, tw - W
    if pad_h > 0 or pad_w > 0:
        pad_top = pad_h // 2
        pad_bot = pad_h - pad_top
        pad_left = pad_w // 2
        pad_right = pad_w - pad_left
        x = F.pad(x, (pad_left, pad_right, pad_top, pad_bot))
    return x


# ---------------------------------------------------------------------------
# Custom weight initialization (variance scaling)
# ---------------------------------------------------------------------------

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
    """Linear layer with variance-scaling initialisation."""
    lin = nn.Linear(in_channels, out_channels)
    variance_scaling_init_(lin.weight, scale=init_scale)
    nn.init.zeros_(lin.bias)
    return lin


def conv2d(in_planes, out_planes, kernel_size=(3, 3), stride=1, dilation=1,
           padding=1, bias=True, padding_mode='zeros', init_scale=1.0):
    """Conv2d with variance-scaling initialisation."""
    conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size,
                     stride=stride, padding=padding, dilation=dilation,
                     bias=bias, padding_mode=padding_mode)
    variance_scaling_init_(conv.weight, scale=init_scale)
    if bias:
        nn.init.zeros_(conv.bias)
    return conv


# ---------------------------------------------------------------------------
# Exponential Moving Average (decoupled from optimizer)
# ---------------------------------------------------------------------------

class EMA:
    """Exponential Moving Average of model parameters, decoupled from the
    optimizer.  Call ``update()`` after each ``optimizer.step()``.

    Compatible with both Adam and RMSprop (or any optimizer).
    """

    def __init__(self, optimizer, ema_decay: float):
        self.ema_decay = float(ema_decay)
        self.apply_ema = self.ema_decay > 0.0
        self.optimizer = optimizer
        self._swap_backup: dict = {}

    # -- public API ----------------------------------------------------------

    def update(self):
        """Update EMA shadow parameters.  Must be called after optimizer.step()."""
        if not self.apply_ema:
            return
        decay = self.ema_decay
        for group in self.optimizer.param_groups:
            for p in group["params"]:
                if not p.requires_grad or p.grad is None:
                    continue
                st = self.optimizer.state[p]
                # Skip if optimizer hasn't initialised this param yet
                if len(st) == 0:
                    continue
                if "ema" not in st:
                    st["ema"] = p.data.detach().clone()
                st["ema"].mul_(decay).add_(p.data, alpha=1.0 - decay)

    def swap_parameters_with_ema(self, store_params_in_ema: bool = True):
        """Replace model weights with EMA weights (for evaluation)."""
        if not self.apply_ema:
            warnings.warn("swap_parameters_with_ema called but EMA is disabled.")
            return
        for group in self.optimizer.param_groups:
            for p in group["params"]:
                if not p.requires_grad:
                    continue
                st = self.optimizer.state[p]
                if "ema" not in st:
                    continue
                if store_params_in_ema:
                    self._swap_backup[id(p)] = p.data.detach().clone()
                p.data.copy_(st["ema"])

    def restore_from_backup(self):
        """Restore original model weights after EMA evaluation."""
        for group in self.optimizer.param_groups:
            for p in group["params"]:
                key = id(p)
                if key in self._swap_backup:
                    p.data.copy_(self._swap_backup[key])
        self._swap_backup.clear()

    def state_dict(self):
        return self.optimizer.state_dict()

    def load_state_dict(self, state_dict):
        self.optimizer.load_state_dict(state_dict)


# ---------------------------------------------------------------------------
# Jacobian norm estimation (for scaled MMD)
# ---------------------------------------------------------------------------

def squared_norm_jacobian_hutch(y: torch.Tensor, x: torch.Tensor,
                                n_hutch: int = 1) -> torch.Tensor:
    """Hutchinson estimator of ||J||_F^2 where J = dy/dx.

    Args:
        y: (B, d) discriminator features (requires grad through x).
        x: (B, C, H, W) input with ``requires_grad=True``.
        n_hutch: number of random projections.

    Returns:
        Scalar: mean ||J||_F^2 over the batch.
    """
    assert x.requires_grad, "x must require grad"
    est = 0.0
    for _ in range(n_hutch):
        v = torch.randn_like(y)
        jtv = torch.autograd.grad(
            outputs=(y * v).sum(), inputs=x,
            create_graph=True, retain_graph=True, only_inputs=True
        )[0]
        est = est + jtv.flatten(1).pow(2).sum(dim=1).mean()
    return est / n_hutch


def squared_norm_jacobian_exact(y: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """Exact computation of mean ||J||_F^2 (loop over output dims)."""
    B, d = y.shape
    assert x.requires_grad, "x must require grad"
    norm2 = torch.zeros(B, device=x.device, dtype=x.dtype)
    for i in range(d):
        grad_i = torch.autograd.grad(
            outputs=y[:, i].sum(), inputs=x,
            create_graph=True, retain_graph=True, only_inputs=True
        )[0]
        norm2 += grad_i.flatten(1).pow(2).sum(dim=1)
    return norm2.mean()


# ---------------------------------------------------------------------------
# Alpha (loss weight) scheduler
# ---------------------------------------------------------------------------

def make_alpha_scheduler(alpha_min: float, alpha_max: float, T: int,
                         kind: str = "cosine"):
    """Return a callable ``get_alpha(step) -> float`` that schedules a loss weight."""
    if kind is None or kind == "none" or alpha_min == alpha_max:
        return lambda step: 1.0

    if kind == "cosine":
        def get_alpha(step: int):
            t = min(max(step, 0) / max(T - 1, 1), 1.0)
            return alpha_min + 0.5 * (alpha_max - alpha_min) * (1.0 - math.cos(math.pi * t))
        return get_alpha

    if kind == "linear":
        def get_alpha(step: int):
            t = min(max(step, 0) / max(T - 1, 1), 1.0)
            return (1.0 - t) * alpha_min + t * alpha_max
        return get_alpha

    raise ValueError(f"Unknown scheduler kind: {kind}")
