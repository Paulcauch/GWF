"""Loss functions for Generative Wasserstein Flows (GWF).

Implements divergence losses (discriminator and generator sides), gradient
penalties, and the transport cost used in the JKO proximal step.

Reference: "A Unifying View of Variational Generative Wasserstein Flows",
           Caucheteux, Bonet, Korba, ICML 2026. Appendix G.
"""

import math
import torch
import torch.nn.functional as F
from mmd import MMD2, scale_MMD


# ============================================================
# Gradient penalties
# ============================================================

def zero_centered_gp(samples: torch.Tensor, critics: torch.Tensor) -> torch.Tensor:
    """Zero-centred gradient penalty (R3GAN style)."""
    grad, = torch.autograd.grad(
        outputs=critics.sum(), inputs=samples, create_graph=True)
    return grad.flatten(1).square().sum(1)


def gradient_penalty(args, d_real, d_fake, real_images, fake_images, netD):
    """Compute gradient penalty based on ``args.gp_type``.

    Supported types:
        - ``"sjko"``:  squared gradient norm w.r.t. real images.
        - ``"interpolated"``:  squared gradient norm on interpolated samples.
        - ``"WGAN"``:  WGAN-GP (penalises deviation from unit norm).
        - ``"r3gan"``:  zero-centred GP on both real and fake.
    """
    if args.gp_type == "sjko":
        grad_real = torch.autograd.grad(
            outputs=d_real.sum(), inputs=real_images, create_graph=True)[0]
        gp = grad_real.view(grad_real.size(0), -1).norm(2, dim=1) ** 2
        return gp

    elif args.gp_type in ("interpolated", "WGAN"):
        batch_size = real_images.size(0)
        # Shape: (B, 1, 1, 1) for images, (B, 1) for 2D data
        alpha_shape = [batch_size] + [1] * (real_images.dim() - 1)
        alpha = torch.rand(alpha_shape, device=real_images.device)
        interpolated = alpha * real_images + (1.0 - alpha) * fake_images
        interpolated.requires_grad_(True)
        prob_interp = netD(interpolated)
        gradients = torch.autograd.grad(
            outputs=prob_interp.sum(), inputs=interpolated,
            create_graph=True, retain_graph=True, only_inputs=True)[0]
        gradients = gradients.view(batch_size, -1)
        grad_norm = torch.sqrt(gradients.pow(2).sum(dim=1) + 1e-12)
        if args.gp_type == "WGAN":
            return (grad_norm - 1.0).pow(2)
        return grad_norm.pow(2)

    elif args.gp_type == "r3gan":
        gp_real = zero_centered_gp(real_images, d_real.mean(dim=1))
        gp_fake = zero_centered_gp(fake_images, d_fake.mean(dim=1))
        return 0.5 * (gp_real + gp_fake)

    raise ValueError(f"Unknown gp_type: {args.gp_type}")


# ============================================================
# Divergence losses — discriminator side
# ============================================================

def loss_D(args, z, real_images, d_real, d_fake, kernel):
    """Compute the discriminator (critic) loss for the chosen divergence.

    The discriminator maximises the divergence estimate, so this function
    returns the *negated* divergence (to be minimised by the optimiser).
    """
    div = args.divergence

    # --- MMD-based ---
    if div == "MMD":
        return -MMD2(d_real, d_fake, kernel, args.mmd_unbiased)
    elif div == "ckMMD":
        return (- kernel(z, d_real) + kernel(z, d_fake)).mean()
    elif div == "iMMD":
        return -MMD2(d_real, d_fake, kernel, args.mmd_unbiased, l=-1)
    elif div == "sMMD":
        scale = scale_MMD(d_real, real_images, args)
        return -scale * MMD2(d_real, d_fake, kernel, args.mmd_unbiased)

    # --- Wasserstein ---
    elif div == "Wasserstein-1":
        return d_fake.mean() - d_real.mean()

    # --- f-divergences ---
    elif div == "KL":
        return (d_fake + torch.exp(-d_real - 1)).mean()
    elif div == "KL_centered":
        return (d_fake + torch.exp(-d_real) - 1).mean()
    elif div == "Shannon":
        return (F.softplus(d_fake) + F.softplus(-d_real)).mean()
    elif div == "chi2":
        return (d_fake + 0.25 * d_real.pow(2) - d_real).mean()
    elif div == "chi2_tight":
        mean_diff = d_fake.mean() - d_real.mean()
        var_real = torch.clamp(d_real.var(unbiased=False), min=1e-4)
        return -(mean_diff.pow(2) / (var_real + 1e-8))
    elif div == "KL_DV":
        return (d_fake + torch.logsumexp(-d_real, dim=0)
                - math.log(d_real.shape[0])).mean()

    raise ValueError(f"Unknown divergence: {div}")


# ============================================================
# Divergence losses — generator side
# ============================================================

def loss_G(args, z, real_images, d_real, d_fake, kernel):
    """Compute the generator loss for the chosen divergence.

    The generator minimises the divergence estimate.
    """
    div = args.divergence

    # --- MMD-based ---
    if div in ("MMD", "iMMD"):
        return MMD2(d_real, d_fake, kernel, unbiased=args.mmd_unbiased)
    elif div == "ckMMD":
        return (-kernel(z, d_fake)).mean()
    elif div == "sMMD":
        scale = scale_MMD(d_real, real_images, args)
        return scale * MMD2(d_real, d_fake, kernel, args.mmd_unbiased)

    # --- Shannon ---
    elif div == "Shannon":
        return F.softplus(-d_fake).mean()

    # --- chi2 tight ---
    elif div == "chi2_tight":
        with torch.no_grad():
            E_nu = d_real.mean()
            V_nu = torch.clamp(d_real.var(unbiased=False), min=1e-4)
        E_T = d_fake.mean()
        return (E_T - E_nu).pow(2) / (V_nu + 1e-8)

    # --- All others: G minimises -E[D(fake)] ---
    elif div in ('KL', 'KL_centered', 'KL_DV', 'chi2',
                 'Wasserstein-1'):
        return -d_fake.mean()

    raise ValueError(f"Unknown divergence: {div}")


# ============================================================
# Symmetric divergence losses  —  D_f(nu || mu) instead of D_f(mu || nu)
# ============================================================
#
# In the original formulation we minimise  F(mu) = D_f(mu || nu)  whose
# variational form is:
#     sup_h { E_mu[h]  -  E_nu[f*(h)] }
#
# In the *symmetric* formulation we minimise  F(mu) = D_f(nu || mu)  whose
# variational form is:
#     sup_h { E_nu[h]  -  E_mu[f*(h)] }
#
# For the D loss this swaps the roles of real and fake.
# For the G loss:  the generator now appears through  -E_mu[f*(h)]  in the
# divergence, so the G loss becomes  -E_fake[f*(h)]  (the f-GAN generator
# loss).
#
# Symmetric divergences (MMD, Wasserstein-1, Jensen-Shannon) are unaffected
# and fall back to the original losses.

def loss_D_sym(args, z, real_images, d_real, d_fake, kernel):
    """Discriminator loss for the *symmetric* divergence D_f(nu || mu).

    For symmetric divergences (MMD, W-1, Shannon) this returns the same
    value as ``loss_D``.  For non-symmetric f-divergences (KL, chi2, …)
    the roles of real / fake embeddings are swapped in the variational form.
    """
    div = args.divergence

    # --- Symmetric divergences: unchanged ---
    if div in ("MMD", "ckMMD", "iMMD", "sMMD", "Wasserstein-1", "Shannon"):
        return loss_D(args, z, real_images, d_real, d_fake, kernel)

    # --- Non-symmetric f-divergences: swap d_real <-> d_fake ---
    if div == "KL":
        return (d_real + torch.exp(-d_fake - 1)).mean()
    elif div == "KL_centered":
        return (d_real + torch.exp(-d_fake) - 1).mean()
    elif div == "chi2":
        return (d_real + 0.25 * d_fake.pow(2) - d_fake).mean()
    elif div == "chi2_tight":
        mean_diff = d_real.mean() - d_fake.mean()
        var_fake = torch.clamp(d_fake.var(unbiased=False), min=1e-4)
        return -(mean_diff.pow(2) / (var_fake + 1e-8))
    elif div == "KL_DV":
        return (d_real + torch.logsumexp(-d_fake, dim=0)
                - math.log(d_fake.shape[0])).mean()

    raise ValueError(f"Unknown divergence: {div}")


def loss_G_sym(args, z, real_images, d_real, d_fake, kernel):
    """Generator loss for the *symmetric* divergence D_f(nu || mu).

    For symmetric divergences this returns the same value as ``loss_G``.
    For non-symmetric f-divergences the generator minimises
    -E_fake[f*(h)], i.e. the conjugate function applied to the fake
    embeddings (this is the standard f-GAN generator loss).

    NOTE: for KL / KL_centered the loss involves an exponential of
    the discriminator output, which may lead to vanishing gradients
    when the discriminator is strong (the well-known "saturating loss"
    issue in f-GANs).
    """
    div = args.divergence

    # --- Symmetric divergences: unchanged ---
    if div in ("MMD", "iMMD", "ckMMD", "sMMD", "Wasserstein-1", "Shannon"):
        return loss_G(args, z, real_images, d_real, d_fake, kernel)

    # --- Non-symmetric f-divergences: G minimises -E_fake[f*(h)] ---
    if div == "KL":
        # f*(s) = exp(s - 1),  h = -d  =>  f*(h) = exp(-d - 1)
        return -(torch.exp(-d_fake - 1)).mean()
    elif div == "KL_centered":
        # f*(s) = exp(s) - 1  =>  f*(-d) = exp(-d) - 1
        return (1 - torch.exp(-d_fake)).mean()
    elif div == "chi2":
        # f*(s) = s + s²/4  =>  f*(-d) = -d + d²/4
        # G loss = -f*(-d_fake) = d_fake - d_fake²/4
        return (d_fake - 0.25 * d_fake.pow(2)).mean()
    elif div == "chi2_tight":
        with torch.no_grad():
            E_real = d_real.mean()
            V_fake = torch.clamp(d_fake.var(unbiased=False), min=1e-4)
        E_fake = d_fake.mean()
        return (E_real - E_fake).pow(2) / (V_fake + 1e-8)
    elif div == "KL_DV":
        # D_KL(nu||mu) depends on G through  -log E_mu[e^h] = -(lse(-d_f) - log N)
        return -(torch.logsumexp(-d_fake, dim=0)
                 - math.log(d_fake.shape[0])).mean()

    raise ValueError(f"Unknown divergence: {div}")


# ============================================================
# Transport cost (JKO proximal term)
# ============================================================

def transport_cost(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Per-sample squared L2 cost averaged over dimensions."""
    return torch.mean((x - y) ** 2, dim=1)
