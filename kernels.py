"""Kernel functions for MMD-based divergences.

Provides both fixed kernels (RBF, energy/Riesz, IMQ, RQ, ...) and a
learnable kernel that is a softmax-weighted mixture of base kernels.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================
# Pairwise distance helpers (Gram matrices)
# ============================================================

def pairwise_l2_sq(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Squared L2 distance matrix.  x: [N,d], y: [M,d] -> [N,M]."""
    x2 = (x * x).sum(dim=1, keepdim=True)          # [N,1]
    y2 = (y * y).sum(dim=1, keepdim=True).t()       # [1,M]
    return (x2 + y2 - 2.0 * (x @ y.t())).clamp_min(0.0)


def pairwise_l2(x: torch.Tensor, y: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    """L2 distance matrix."""
    return torch.sqrt(pairwise_l2_sq(x, y) + eps)


def pairwise_l1(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """L1 distance matrix."""
    return (x[:, None, :] - y[None, :, :]).abs().sum(dim=-1)


# ============================================================
# Aligned distance helpers (diagonal, same batch)
# ============================================================

def aligned_l2_sq(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Squared L2 distance, aligned pairs.  x: [B,d], y: [B,d] -> [B]."""
    return ((x - y) ** 2).sum(dim=-1).clamp_min(0.0)


def aligned_l2(x: torch.Tensor, y: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    return torch.sqrt(aligned_l2_sq(x, y) + eps)


def aligned_l1(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    return (x - y).abs().sum(dim=-1)


# ============================================================
# Base kernels (support both "gram" and "diag" modes)
# ============================================================

def gaussian_kernel(x, y, sigma=10.0, mode="gram"):
    d2 = pairwise_l2_sq(x, y) if mode == "gram" else aligned_l2_sq(x, y)
    return torch.exp(-d2 / (2.0 * sigma * sigma))


def rbf_mixture_kernel(x, y, K=5, mode="gram"):
    """Sum of Gaussian kernels with geometrically spaced bandwidths."""
    out = 0.0
    sigma_q = 0.5
    for _ in range(K):
        sigma_q *= 2.0
        out = out + gaussian_kernel(x, y, sigma=sigma_q, mode=mode)
    return out


def laplacian_kernel(x, y, sigma=100.0, mode="gram"):
    d1 = pairwise_l1(x, y) if mode == "gram" else aligned_l1(x, y)
    return torch.exp(-d1 / sigma)


def exponential_kernel(x, y, sigma=10.0, mode="gram"):
    d = pairwise_l2(x, y) if mode == "gram" else aligned_l2(x, y)
    return torch.exp(-d / sigma)


def matern_3_2_kernel(x, y, alpha=1.0, l=10.0, mode="gram"):
    r = (pairwise_l2(x, y) if mode == "gram" else aligned_l2(x, y)) / l
    sqrt3 = math.sqrt(3.0)
    return alpha * (1.0 + sqrt3 * r) * torch.exp(-sqrt3 * r)


def matern_5_2_kernel(x, y, alpha=1.0, l=10.0, mode="gram"):
    r = (pairwise_l2(x, y) if mode == "gram" else aligned_l2(x, y)) / l
    sqrt5 = math.sqrt(5.0)
    return alpha * (1.0 + sqrt5 * r + (5.0 / 3.0) * r * r) * torch.exp(-sqrt5 * r)


def riesz_psd_kernel(x, y, eps=1e-6, mode="gram"):
    """Positive semi-definite Riesz kernel: -||x-y|| + ||x|| + ||y||."""
    if mode == "gram":
        d_xy = pairwise_l2(x, y, eps=eps)
        nx = torch.sqrt((x ** 2).sum(dim=1, keepdim=True) + eps)
        ny = torch.sqrt((y ** 2).sum(dim=1, keepdim=True) + eps)
        return -d_xy + nx + ny.T
    else:
        d_xy = aligned_l2(x, y, eps=eps)
        nx = torch.sqrt((x ** 2).sum(dim=1) + eps)
        ny = torch.sqrt((y ** 2).sum(dim=1) + eps)
        return -d_xy + nx + ny


# ============================================================
# Additional fixed kernels (Gram mode only)
# ============================================================

def linear_kernel(x, y):
    return x @ y.t()


def energy_kernel(x, y, eps=1e-6):
    """Negative distance kernel: K(x,y) = -||x-y||."""
    return -pairwise_l2(x, y, eps=eps)


def energy_kernel_psd(x, y, eps=1e-6):
    """PSD centred energy kernel: K(x,y) - K(x,0) - K(0,y)."""
    K = -pairwise_l2(x, y, eps=eps)
    K_x0 = -torch.sqrt((x * x).sum(dim=1, keepdim=True) + eps)
    K_0y = -torch.sqrt((y * y).sum(dim=1, keepdim=True).t() + eps)
    return K - K_x0 - K_0y


def imq_kernel(x, y, cste=1.0, beta=-0.5):
    """Inverse multi-quadratic kernel."""
    d2 = pairwise_l2_sq(x, y)
    return (cste ** 2 + d2).pow(beta)


def rq_kernel(x, y, alpha=1.0, eps=1e-12):
    """Rational quadratic kernel."""
    d2 = pairwise_l2_sq(x, y)
    return (1.0 + d2 / (2.0 * alpha + eps)).pow(-alpha)


def mix_rq_kernel(x, y, alphas=(0.1, 1.0, 10.0), wts=None, eps=1e-12):
    """Mixture of rational quadratic kernels."""
    if wts is None:
        wts = [1.0] * len(alphas)
    d2 = pairwise_l2_sq(x, y)
    K = 0.0
    for alpha, wt in zip(alphas, wts):
        K = K + wt * (1.0 + d2 / (2.0 * alpha + eps)).pow(-alpha)
    return K


# ============================================================
# Learned kernel (softmax-weighted mixture of base kernels)
# ============================================================

class _LearnedKernelBase(nn.Module):
    """Base class for learnable kernel mixtures.

    Inspired by CKGAN (Zhang et al., 2025):
    
    Learns a weighted combination k(x,y) = sum_i w_i k_i(x,y) where the
    weights w_i >= 0 are optimised jointly with the generator and discriminator.
    """

    def __init__(self, is_use_softmax=True, is_use_one_hot=False):
        super().__init__()
        self.is_use_softmax = is_use_softmax
        self.is_use_one_hot = is_use_one_hot
        self.kernel_weight = nn.Parameter(torch.zeros(7))

    def _weights(self):
        w = self.kernel_weight
        if self.is_use_softmax:
            w = F.softmax(w, dim=0)
            if self.is_use_one_hot:
                idx = torch.argmax(w)
                one_hot = torch.zeros_like(w)
                one_hot[idx] = 1.0
                w = w * one_hot
        return w

    def _mix(self, x, y, mode: str):
        w = self._weights()
        K = 0.0
        K = K + w[0] * gaussian_kernel(x, y, sigma=10.0, mode=mode)
        K = K + w[1] * rbf_mixture_kernel(x, y, K=5, mode=mode)
        K = K + w[2] * laplacian_kernel(x, y, sigma=100.0, mode=mode)
        K = K + w[3] * exponential_kernel(x, y, sigma=10.0, mode=mode)
        K = K + w[4] * matern_3_2_kernel(x, y, alpha=1.0, l=10.0, mode=mode)
        K = K + w[5] * matern_5_2_kernel(x, y, alpha=1.0, l=10.0, mode=mode)
        K = K + w[6] * riesz_psd_kernel(x, y, mode=mode)
        return K


class LearnedKernelDiag(_LearnedKernelBase):
    """Learned kernel for aligned pairs: x:[B,d], y:[B,d] -> [B]."""
    def forward(self, x, y):
        return self._mix(x, y, mode="diag")


class LearnedKernelGram(_LearnedKernelBase):
    """Learned kernel producing Gram matrix: x:[N,d], y:[M,d] -> [N,M]."""
    def forward(self, x, y):
        return self._mix(x, y, mode="gram")


# ============================================================
# Kernel factory
# ============================================================

def get_kernel(args):
    """Return a kernel function or nn.Module based on ``args.kernel``."""
    k = args.kernel

    if k == "linear":
        return linear_kernel
    if k == "rbf":
        return lambda x, y: gaussian_kernel(x, y, sigma=args.sigma_kernel, mode="gram")
    if k == "riesz":
        return energy_kernel
    if k == "riesz_psd":
        return energy_kernel_psd
    if k == "imq":
        return imq_kernel
    if k == "rq":
        return rq_kernel
    if k == "mix_rq":
        return mix_rq_kernel
    if k == "learned_kernel":
        if args.divergence == "ckMMD":
            return LearnedKernelDiag(is_use_softmax=True, is_use_one_hot=False)
        else:
            return LearnedKernelGram(is_use_softmax=True, is_use_one_hot=False)

    raise ValueError(f"Unknown kernel: {k}")
