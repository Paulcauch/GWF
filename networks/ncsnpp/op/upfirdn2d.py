# ---------------------------------------------------------------
# Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
# ---------------------------------------------------------------
"""
Originated from https://github.com/rosinality/stylegan2-pytorch
The license for the original version of this file can be found in this directory (LICENSE_MIT).
"""

import os
from pathlib import Path
from collections import abc

import torch
from torch.nn import functional as F
from torch.autograd import Function

# On n'importera load/CUDA_HOME que si on tente vraiment la compilation
try:
    from torch.utils.cpp_extension import load, CUDA_HOME  # type: ignore
except Exception:  # torch sans cpp_extension ou env restreint
    load, CUDA_HOME = None, None  # fallback hard

module_path = os.path.dirname(__file__)

# --- Décision: compile l'extension uniquement si CUDA est dispo ET non forcé en CPU.
USE_CUDA_EXT = (
    torch.cuda.is_available()
    and (CUDA_HOME is not None)
    and (os.environ.get("FORCE_CPU_UPFIRDN2D", "0") != "1")
)

upfirdn2d_op = None
if USE_CUDA_EXT and load is not None:
    # On inclut le .cu uniquement si CUDA est vraiment dispo
    _sources = [
        os.path.join(module_path, "upfirdn2d.cpp"),
        os.path.join(module_path, "upfirdn2d_kernel.cu"),
    ]
    try:
        upfirdn2d_op = load(
            name="upfirdn2d",
            sources=_sources,
            extra_cflags=["-O3", "-std=c++17"],
            extra_cuda_cflags=["-O3"],
            verbose=False,
        )
    except Exception as e:
        print("upfirdn2d: extension CUDA désactivée, fallback PyTorch. Raison:", e)
        upfirdn2d_op = None
else:
    # Pas de CUDA (Mac, MPS, CPU…) ou fallback forcé
    print("upfirdn2d: utilisation du fallback PyTorch (pas de CUDA extension).")


class UpFirDn2dBackward(Function):
    @staticmethod
    def forward(
        ctx, grad_output, kernel, grad_kernel, up, down, pad, g_pad, in_size, out_size
    ):
        # Ce chemin n'est utilisé que si l'extension est compilée (CUDA)
        up_x, up_y = up
        down_x, down_y = down
        g_pad_x0, g_pad_x1, g_pad_y0, g_pad_y1 = g_pad

        grad_output = grad_output.reshape(-1, out_size[0], out_size[1], 1)

        grad_input = upfirdn2d_op.upfirdn2d(
            grad_output,
            grad_kernel,
            down_x,
            down_y,
            up_x,
            up_y,
            g_pad_x0,
            g_pad_x1,
            g_pad_y0,
            g_pad_y1,
        )
        grad_input = grad_input.view(in_size[0], in_size[1], in_size[2], in_size[3])

        ctx.save_for_backward(kernel)

        pad_x0, pad_x1, pad_y0, pad_y1 = pad

        ctx.up_x = up_x
        ctx.up_y = up_y
        ctx.down_x = down_x
        ctx.down_y = down_y
        ctx.pad_x0 = pad_x0
        ctx.pad_x1 = pad_x1
        ctx.pad_y0 = pad_y0
        ctx.pad_y1 = pad_y1
        ctx.in_size = in_size
        ctx.out_size = out_size

        return grad_input

    @staticmethod
    def backward(ctx, gradgrad_input):
        kernel, = ctx.saved_tensors

        gradgrad_input = gradgrad_input.reshape(-1, ctx.in_size[2], ctx.in_size[3], 1)

        gradgrad_out = upfirdn2d_op.upfirdn2d(
            gradgrad_input,
            kernel,
            ctx.up_x,
            ctx.up_y,
            ctx.down_x,
            ctx.down_y,
            ctx.pad_x0,
            ctx.pad_x1,
            ctx.pad_y0,
            ctx.pad_y1,
        )
        gradgrad_out = gradgrad_out.view(
            ctx.in_size[0], ctx.in_size[1], ctx.out_size[0], ctx.out_size[1]
        )

        return gradgrad_out, None, None, None, None, None, None, None, None


class UpFirDn2d(Function):
    @staticmethod
    def forward(ctx, input, kernel, up, down, pad):
        # Ce chemin n'est utilisé que si l'extension est compilée (CUDA)
        up_x, up_y = up
        down_x, down_y = down
        pad_x0, pad_x1, pad_y0, pad_y1 = pad

        kernel_h, kernel_w = kernel.shape
        batch, channel, in_h, in_w = input.shape
        ctx.in_size = input.shape

        input = input.reshape(-1, in_h, in_w, 1)
        ctx.save_for_backward(kernel, torch.flip(kernel, [0, 1]))

        out_h = (in_h * up_y + pad_y0 + pad_y1 - kernel_h) // down_y + 1
        out_w = (in_w * up_x + pad_x0 + pad_x1 - kernel_w) // down_x + 1
        ctx.out_size = (out_h, out_w)

        ctx.up = (up_x, up_y)
        ctx.down = (down_x, down_y)
        ctx.pad = (pad_x0, pad_x1, pad_y0, pad_y1)

        g_pad_x0 = kernel_w - pad_x0 - 1
        g_pad_y0 = kernel_h - pad_y0 - 1
        g_pad_x1 = in_w * up_x - out_w * down_x + pad_x0 - up_x + 1
        g_pad_y1 = in_h * up_y - out_h * down_y + pad_y0 - up_y + 1

        ctx.g_pad = (g_pad_x0, g_pad_x1, g_pad_y0, g_pad_y1)

        out = upfirdn2d_op.upfirdn2d(
            input, kernel, up_x, up_y, down_x, down_y, pad_x0, pad_x1, pad_y0, pad_y1
        )
        out = out.view(-1, channel, out_h, out_w)

        return out

    @staticmethod
    def backward(ctx, grad_output):
        kernel, grad_kernel = ctx.saved_tensors

        grad_input = UpFirDn2dBackward.apply(
            grad_output,
            kernel,
            grad_kernel,
            ctx.up,
            ctx.down,
            ctx.pad,
            ctx.g_pad,
            ctx.in_size,
            ctx.out_size,
        )

        return grad_input, None, None, None, None


# --------- API publique, avec fallback propre ---------

def upfirdn2d(input, kernel, up=1, down=1, pad=(0, 0)):
    """
    Appelle l’extension CUDA si elle est dispo ET que le tenseur est CUDA,
    sinon utilise le fallback PyTorch (autograd-friendly) qui marche sur CPU/MPS.
    """
    use_ext = (upfirdn2d_op is not None) and input.is_cuda
    if not use_ext:
        return upfirdn2d_native(
            input, kernel, up, up, down, down, pad[0], pad[1], pad[0], pad[1]
        )
    else:
        return UpFirDn2d.apply(
            input, kernel, (up, up), (down, down), (pad[0], pad[1], pad[0], pad[1])
        )


def upfirdn2d_ada(input, kernel, up=1, down=1, pad=(0, 0)):
    if not isinstance(up, abc.Iterable):
        up = (up, up)

    if not isinstance(down, abc.Iterable):
        down = (down, down)

    if len(pad) == 2:
        pad = (pad[0], pad[1], pad[0], pad[1])

    use_ext = (upfirdn2d_op is not None) and input.is_cuda
    if not use_ext:
        return upfirdn2d_native(input, kernel, *up, *down, *pad)
    else:
        return UpFirDn2d.apply(input, kernel, up, down, pad)


def upfirdn2d_native(
    input, kernel, up_x, up_y, down_x, down_y, pad_x0, pad_x1, pad_y0, pad_y1
):
    """
    Fallback PyTorch pur (fonctionne sur CPU/MPS/CUDA, autograd OK).
    """
    # S’assurer que tout est sur le même device/dtype
    device = input.device
    dtype = input.dtype
    kernel = kernel.to(device=device, dtype=dtype)

    _, channel, in_h, in_w = input.shape
    input = input.reshape(-1, in_h, in_w, 1)

    _, in_h, in_w, minor = input.shape
    kernel_h, kernel_w = kernel.shape

    out = input.view(-1, in_h, 1, in_w, 1, minor)
    # Upsample par insertion de zéros (équivalent à nearest puis conv FIR)
    out = F.pad(out, [0, 0, 0, up_x - 1, 0, 0, 0, up_y - 1])
    out = out.view(-1, in_h * up_y, in_w * up_x, minor)

    # Padding explicite avant convolution
    out = F.pad(
        out, [0, 0, max(pad_x0, 0), max(pad_x1, 0), max(pad_y0, 0), max(pad_y1, 0)]
    )
    out = out[
        :,
        max(-pad_y0, 0) : out.shape[1] - max(-pad_y1, 0),
        max(-pad_x0, 0) : out.shape[2] - max(-pad_x1, 0),
        :,
    ]

    out = out.permute(0, 3, 1, 2)
    out = out.reshape(
        [-1, 1, in_h * up_y + pad_y0 + pad_y1, in_w * up_x + pad_x0 + pad_x1]
    )

    # Filtre FIR (partagé sur les canaux) — on place le kernel sur le bon device/dtype
    w = torch.flip(kernel, [0, 1]).to(device=device, dtype=dtype).view(
        1, 1, kernel_h, kernel_w
    )
    out = F.conv2d(out, w)

    out = out.reshape(
        -1,
        minor,
        in_h * up_y + pad_y0 + pad_y1 - kernel_h + 1,
        in_w * up_x + pad_x0 + pad_x1 - kernel_w + 1,
    )
    out = out.permute(0, 2, 3, 1)

    # Downsample par sous-échantillonnage
    out = out[:, ::down_y, ::down_x, :]

    out_h = (in_h * up_y + pad_y0 + pad_y1 - kernel_h) // down_y + 1
    out_w = (in_w * up_x + pad_x0 + pad_x1 - kernel_w) // down_x + 1

    return out.view(-1, channel, out_h, out_w)
