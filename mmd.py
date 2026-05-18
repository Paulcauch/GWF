"""Maximum Mean Discrepancy (MMD) computation and scaling."""

import torch
from kernels import energy_kernel
from utils import squared_norm_jacobian_exact, squared_norm_jacobian_hutch


def MMD2(x: torch.Tensor, y: torch.Tensor, kernel=energy_kernel,
         unbiased: bool = False, l: float = 1.0) -> torch.Tensor:
    """Compute (biased or unbiased) MMD^2 between samples x and y.

    Args:
        x: (N, d) samples from distribution P.
        y: (M, d) samples from distribution Q.
        kernel: callable ``kernel(x, y) -> (N, M)`` Gram matrix.
        unbiased: if True, use the U-statistic estimator.
        l: interpolation coefficient.  ``l = 1`` gives the standard MMD.
           ``l = -1`` drops the cross-term (used for iMMD discriminator loss).

    Returns:
        Scalar MMD^2 estimate.
    """
    kxx = kernel(x, x)
    kyy = kernel(y, y)
    kxy = kernel(x, y) if l > -1 else None

    if unbiased:
        N, M = x.shape[0], y.shape[0]
        kxx_off = kxx - torch.diag(torch.diag(kxx))
        kyy_off = kyy - torch.diag(torch.diag(kyy))
        term_xx = kxx_off.sum() / (N * (N - 1))
        term_yy = kyy_off.sum() / (M * (M - 1))
        term_xy = kxy.sum() / (N * M) if kxy is not None else 0.0
    else:
        term_xx = kxx.mean()
        term_yy = kyy.mean()
        term_xy = kxy.mean() if kxy is not None else 0.0

    return l * term_xx + term_yy - (l + 1) * term_xy


def scale_MMD(encode_y: torch.Tensor, y: torch.Tensor, args) -> torch.Tensor:
    """Compute scaling factor for scaled MMD (sMMD).

    Uses the Jacobian norm of the encoder to down-weight the MMD when the
    discriminator has large gradients, providing implicit regularisation.

    Args:
        encode_y: (B, d) discriminator embeddings of real images.
        y: (B, C, H, W) real images (must have ``requires_grad=True``).
        args: must contain ``embed_dim``, ``n_hutch``, ``scaling_variant``,
              ``scaling_coeff``.

    Returns:
        Scalar scaling factor in (0, 1].
    """
    if args.embed_dim <= 16:
        norm2_jac = squared_norm_jacobian_exact(encode_y, y)
    else:
        norm2_jac = squared_norm_jacobian_hutch(encode_y, y, n_hutch=args.n_hutch)

    if args.scaling_variant == "grad":
        scale = 1.0 / (1.0 + args.scaling_coeff * norm2_jac)
    elif args.scaling_variant == "value_and_grad":
        norm_disc = encode_y.pow(2).mean()
        scale = 1.0 / (1.0 + args.scaling_coeff * (norm2_jac + norm_disc))
    else:
        scale = 1.0

    return scale
