"""Generator and discriminator architectures for Generative Wasserstein Flows.

Supported architectures (via ``--model_name``):
    - ``resnet_MMDGAN``:  ResNet G + ResNet D (ResNet MMD-GAN in paper).
    - ``ncsnpp``:         NCSN++ generator + small discriminator (scalar output).
    - ``ncsnpp_embed``:   NCSN++ generator + embedding discriminator (Large-Net in paper).
    - ``otm``:            OTM generator + discriminator (Small-Net in paper).
    - ``unet``:           UNet generator + discriminator (U-Net in paper).
    - ``mlp_2d``:         Simple MLP for 2D toy datasets.

Reference: "A Unifying View of Variational Generative Wasserstein Flows",
           Caucheteux, Bonet, Korba, 2026.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.parametrizations import spectral_norm

from .unet import UNet
from dataset import get_size_dataset


# =====================================================================
# Helpers
# =====================================================================

def maybe_sn(layer: nn.Module, use_sn: bool) -> nn.Module:
    return spectral_norm(layer) if use_sn else layer


# =====================================================================
# 2D MLP Generator and Discriminator (mlp_2d)
# =====================================================================

class Generator2D(nn.Module):
    """MLP generator for 2D toy datasets: z -> MLP -> R^2."""
    def __init__(self, z_dim=2, hidden_dim=256, out_dim=2):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(z_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, z):
        return self.model(z)


class Discriminator2D(nn.Module):
    """MLP discriminator for 2D toy datasets: R^2 -> R^embed_dim."""
    def __init__(self, in_dim=2, hidden_dim=256, embed_dim=32):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, embed_dim),
        )

    def forward(self, x):
        return self.model(x)


# =====================================================================
# ResNet-based Generator and Discriminator (resnet_MMDGAN)
# =====================================================================

class GResBlock(nn.Module):
    """Generator residual block with nearest-neighbour upsampling."""
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.bn1 = nn.BatchNorm2d(in_ch)
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, 1, 1)
        self.bn2 = nn.BatchNorm2d(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, 1, 1)
        self.skip_conv = nn.Conv2d(in_ch, out_ch, 1, 1, 0)

    def forward(self, x):
        skip = x
        out = F.relu(self.bn1(x), inplace=False)
        out = F.interpolate(out, scale_factor=2, mode="nearest")
        out = self.conv1(out)
        out = F.relu(self.bn2(out), inplace=False)
        out = self.conv2(out)
        skip = F.interpolate(skip, scale_factor=2, mode="nearest")
        skip = self.skip_conv(skip)
        return out + skip


class GeneratorResNet(nn.Module):
    """ResNet generator: z -> Linear -> reshape -> N upsample blocks -> BN/ReLU -> conv -> tanh."""
    def __init__(self, z_dim, out_channels, start_res, base_ch=256, n_up=2):
        super().__init__()
        self.start_res = start_res
        self.base_ch = base_ch
        self.fc = nn.Linear(z_dim, base_ch * start_res * start_res)
        self.blocks = nn.Sequential(*[GResBlock(base_ch, base_ch) for _ in range(n_up)])
        self.bn_final = nn.BatchNorm2d(base_ch)
        self.conv_out = nn.Conv2d(base_ch, out_channels, 3, 1, 1)

    def forward(self, z):
        B = z.size(0)
        z = z.reshape(B, -1)
        x = self.fc(z).view(B, self.base_ch, self.start_res, self.start_res)
        x = self.blocks(x)
        x = F.relu(self.bn_final(x), inplace=False)
        return torch.tanh(self.conv_out(x))


class DFirstResBlock(nn.Module):
    """First discriminator res block (no pre-activation)."""
    def __init__(self, in_ch, out_ch, down, use_sn):
        super().__init__()
        self.down = down
        self.conv1 = maybe_sn(nn.Conv2d(in_ch, out_ch, 3, 1, 1), use_sn)
        self.conv2 = maybe_sn(nn.Conv2d(out_ch, out_ch, 3, 1, 1), use_sn)
        self.skip_conv = maybe_sn(nn.Conv2d(in_ch, out_ch, 1, 1, 0), use_sn)
        self.avgpool = nn.AvgPool2d(2, 2) if down else nn.Identity()

    def forward(self, x):
        skip = x
        out = self.conv1(x)
        out = F.relu(out, inplace=False)
        out = self.conv2(out)
        if self.down:
            out = self.avgpool(out)
            skip = self.avgpool(self.skip_conv(skip))
        return out + skip


class DResBlock(nn.Module):
    """Discriminator residual block with optional downsampling."""
    def __init__(self, in_ch, out_ch, down, use_sn):
        super().__init__()
        self.down = down
        self.conv1 = maybe_sn(nn.Conv2d(in_ch, out_ch, 3, 1, 1), use_sn)
        self.conv2 = maybe_sn(nn.Conv2d(out_ch, out_ch, 3, 1, 1), use_sn)
        self.skip_conv = maybe_sn(nn.Conv2d(in_ch, out_ch, 1, 1, 0), use_sn)
        self.avgpool = nn.AvgPool2d(2, 2) if down else nn.Identity()

    def forward(self, x):
        skip = x
        out = F.relu(x, inplace=False)
        out = self.conv1(out)
        out = F.relu(out, inplace=False)
        out = self.conv2(out)
        if self.down:
            out = self.avgpool(out)
            skip = self.avgpool(self.skip_conv(skip))
        else:
            skip = self.skip_conv(skip)
        return out + skip


class DiscriminatorResNet(nn.Module):
    """ResNet discriminator: down-blocks -> flatten -> Linear -> embed_dim."""
    def __init__(self, img_size, in_channels, embed_dim, use_sn,
                 base_ch=128, n_down=2, n_nodown=1):
        super().__init__()
        assert n_down >= 1
        blocks = [DFirstResBlock(in_channels, base_ch, down=True, use_sn=use_sn)]
        blocks += [DResBlock(base_ch, base_ch, down=True, use_sn=use_sn)
                   for _ in range(n_down - 1)]
        blocks += [DResBlock(base_ch, base_ch, down=False, use_sn=use_sn)
                   for _ in range(n_nodown)]
        self.blocks = nn.Sequential(*blocks)
        feat_res = img_size // (2 ** n_down)
        self.fc = maybe_sn(nn.Linear(base_ch * feat_res * feat_res, embed_dim), use_sn)

    def forward(self, x):
        x = self.blocks(x)
        x = F.relu(x, inplace=False)
        return self.fc(torch.flatten(x, 1))


# =====================================================================
# NCSN++ configuration helper
# =====================================================================

def apply_ncsnpp_args(args):
    """Set NCSN++ architecture hyperparameters on the args namespace."""
    args.centered = True
    args.num_channels_dae = 128
    args.n_mlp = 4
    args.ngf = 64
    args.ch_mult = [1, 2, 2, 2]
    args.num_res_blocks = 2
    args.attn_resolutions = (16,)
    args.dropout = 0.0
    args.resamp_with_conv = True
    args.fir = True
    args.fir_kernel = [1, 3, 3, 1]
    args.skip_rescale = True
    args.resblock_type = "biggan"
    args.progressive = "none"
    args.progressive_input = "residual"
    args.progressive_combine = "sum"
    args.embedding_type = "positional"
    args.fourier_scale = 16.0
    args.not_use_tanh = args.images_in_0_1
    args.z_emb_dim = 256
    args.use_sigmoid = args.images_in_0_1


# =====================================================================
# OTM configuration helper
# =====================================================================

def apply_otm_args(args):
    """Set OTM architecture hyperparameters on the args namespace."""
    args.imageSize = args.image_size
    args.G_conv = 'convT'
    args.G_normalization = "BN"
    args.G_activation = "relu"
    args.G_linear = "linear"
    args.G_bias = False
    args.D_conv = "conv"
    args.D_normalization = "BN"
    args.D_activation = "lrelu"
    args.D_linear = "linear"
    args.D_bias = False
    args.projection_dim = 128
    args.conditioning = "concat"
    args.num_classes = 1
    args.nc = args.num_channels
    if args.imageSize < 32:
        args.ngf = 40


# =====================================================================
# UNet discriminator configuration helper
# =====================================================================

def apply_unet_discriminator_args(args):
    """Set discriminator hyperparameters for the UNet generator."""
    args.D_linear = "spectral_linear"
    args.nc = args.num_channels
    args.conditioning = "concat"
    args.num_classes = 1
    args.projection_dim = 128
    args.D_conv = "spectral_conv"
    args.D_activation = "lrelu"
    args.D_bias = False


# =====================================================================
# Positivity wrapper for chi2_tight critic
# =====================================================================

class CriticWithPos(nn.Module):
    """Wraps a discriminator to ensure non-negative output (for chi2_tight)."""
    def __init__(self, base, mode="softplus"):
        super().__init__()
        self.base = base
        self.mode = mode

    def forward(self, x):
        out = self.base(x)
        return F.softplus(out) if self.mode == "softplus" else F.relu(out)


# =====================================================================
# Model factory
# =====================================================================

def get_model(args):
    """Instantiate generator T and discriminator D based on ``args.model_name``.

    Returns:
        (T, D): generator and discriminator modules.
    """
    name = args.model_name

    if name == "resnet_MMDGAN":
        nz = args.nz
        if args.dataset in ("mnist", "mnist28"):
            T = GeneratorResNet(z_dim=nz, out_channels=args.num_channels,
                                start_res=7, base_ch=256, n_up=2)
            D = DiscriminatorResNet(img_size=args.image_size, in_channels=args.num_channels,
                                    embed_dim=args.embed_dim, use_sn=args.use_sn,
                                    base_ch=128, n_down=2, n_nodown=1)
        elif args.dataset == "cifar10":
            T = GeneratorResNet(z_dim=nz, out_channels=args.num_channels,
                                start_res=4, base_ch=256, n_up=3)
            D = DiscriminatorResNet(img_size=args.image_size, in_channels=args.num_channels,
                                    embed_dim=args.embed_dim, use_sn=args.use_sn,
                                    base_ch=128, n_down=2, n_nodown=2)
        elif args.dataset in ("celeba", "lsun"):
            if args.image_size < 64 or (args.image_size & (args.image_size - 1)) != 0:
                raise ValueError(
                    "resnet_MMDGAN: for celeba/lsun, image_size must be a power of 2 and >= 64 "
                    f"(got {args.image_size})"
                )
            n_up = int(math.log2(args.image_size)) - 2
            n_down = int(math.log2(args.image_size)) - 3
            T = GeneratorResNet(z_dim=nz, out_channels=args.num_channels,
                                start_res=4, base_ch=256, n_up=n_up)
            D = DiscriminatorResNet(img_size=args.image_size, in_channels=args.num_channels,
                                    embed_dim=args.embed_dim, use_sn=args.use_sn,
                                    base_ch=128, n_down=n_down, n_nodown=2)
        else:
            raise ValueError(f"resnet_MMDGAN: unsupported dataset {args.dataset}")

    elif name == "ncsnpp":
        from .ncsnpp.discriminator import Discriminator_small, Discriminator_large
        from .ncsnpp.ncsnpp_generator_adagn import NCSNpp
        apply_ncsnpp_args(args)
        T = NCSNpp(args)
        if args.image_size <= 32:
            D = Discriminator_small(nc=args.num_channels, ngf=args.ngf,
                                    act=nn.LeakyReLU(0.2))
        else:
            D = Discriminator_large(image_size=args.image_size, nc=args.num_channels,
                                    ngf=args.ngf, act=nn.LeakyReLU(0.2))

    elif name == "ncsnpp_embed":
        from .ncsnpp.discriminator import Discriminator_small_embed, Discriminator_large_embed
        from .ncsnpp.ncsnpp_generator_adagn import NCSNpp
        apply_ncsnpp_args(args)
        T = NCSNpp(args)
        if args.image_size <= 32:
            D = Discriminator_small_embed(nc=args.num_channels, ngf=args.ngf,
                                          o_dim=args.embed_dim, act=nn.LeakyReLU(0.2))
        else:
            D = Discriminator_large_embed(image_size=args.image_size, nc=args.num_channels,
                                          ngf=args.ngf, o_dim=args.embed_dim,
                                          act=nn.LeakyReLU(0.2))

    elif name == "otm":
        from .modelOTM.OTM import Generator_otm, Discriminator_otm
        apply_otm_args(args)
        T = Generator_otm(args)
        D = Discriminator_otm(args)

    elif name == "unet":
        from .modelOTM.OTM import Discriminator_unet
        apply_unet_discriminator_args(args)
        if args.dataset == "cifar10":
            args.ndf = 164
            T = UNet(input_channels=3, input_height=32, ch=96,
                     ch_mult=(1, 2, 2, 2), num_res_blocks=2,
                     attn_resolutions=(16,), resamp_with_conv=True, dropout=0.0)
        elif args.dataset in ("mnist", "mnist28"):
            args.ndf = 128
            T = UNet(input_channels=1, input_height=28, ch=96,
                     ch_mult=(1, 2, 2), num_res_blocks=2,
                     attn_resolutions=(), resamp_with_conv=True, dropout=0.0)
        else:
            raise ValueError(f"unet: unsupported dataset {args.dataset}")
        D = Discriminator_unet(args)

    elif name == "mlp_2d":
        T = Generator2D(z_dim=args.nz, hidden_dim=128, out_dim=2)
        D = Discriminator2D(in_dim=2, hidden_dim=128, embed_dim=args.embed_dim)

    else:
        raise ValueError(f"Unknown model_name: {name}")

    if args.divergence == "chi2_tight":
        D = CriticWithPos(D, mode="softplus")

    return T, D
