"""Training script for Generative Wasserstein Flows (GWF).

Implements the JKO (Jordan-Kinderlehrer-Otto) proximal scheme for generative
modelling, as described in:

    "A Unifying View of Variational Generative Wasserstein Flows"
    Paul Caucheteux, Clement Bonet, Anna Korba, ICML 2026.

At each outer (JKO) step, the generator solves the proximal problem:
    T_{k+1} = argmin_T  D(T#mu_0 || nu) + (1/2tau) * ||T - T_k||^2_{L2(mu_0)}
via alternating optimisation of a discriminator h and generator T.

Supports f-divergences (KL, JS, chi2), IPMs (Wasserstein-1), and squared MMD.
The --symmetric flag switches to D_f(nu || mu) (standard f-GAN direction).

Usage:
    python train.py --dataset cifar10 --divergence MMD --JKO --model_name resnet_MMDGAN ...

See ``--help`` for all options and ``scripts/`` for example configurations.
"""

import argparse
import os
import json
import shutil
import copy
from datetime import datetime, timedelta

import torch
import torch.optim as optim
import torch.nn.functional as F
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import torchvision
from torchvision.utils import save_image
from pytorch_fid.fid_score import calculate_fid_given_paths

from networks import get_model
from mmd import MMD2
from kernels import get_kernel
from dataset import (get_sampler, get_noise, get_noise_dataset,
                     get_size_entire_training_dataset, is_2d_dataset)
from utils import EMA, ensure_dir, to_0_1, make_alpha_scheduler, save_2d_scatter
from eval import (dump_real_images_if_needed, generate_fake_dir,
                  compute_inception_score_from_dir, compute_kid_from_dirs,
                  append_metric, save_metrics_json, save_ckpt, load_ckpt)
from losses import (loss_D, loss_G, loss_D_sym, loss_G_sym,
                    gradient_penalty, transport_cost)


# ============================================================
# Training loop
# ============================================================

def train(args):
    use_ddp = args.ddp and int(os.environ.get("WORLD_SIZE", "1")) > 1
    rank, local_rank, world_size = 0, 0, 1
    if use_ddp:
        if not torch.cuda.is_available():
            raise RuntimeError("DDP requires CUDA GPUs.")
        if not dist.is_initialized():
            dist.init_process_group(
                backend="nccl",
                init_method="env://",
                timeout=timedelta(hours=2),
            )
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        local_rank = int(os.environ.get("LOCAL_RANK", rank))
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    is_main = rank == 0

    def unwrap_model(module):
        return module.module if isinstance(module, DDP) else module

    def ddp_barrier():
        if use_ddp:
            dist.barrier(device_ids=[local_rank])

    # Seeds
    seed = args.seed + rank
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    is_2d = is_2d_dataset(args.dataset)

    # ----- Loss direction (symmetric vs forward) -----
    if args.symmetric:
        compute_D_loss = loss_D_sym
        compute_G_loss = loss_G_sym
    else:
        compute_D_loss = loss_D
        compute_G_loss = loss_G

    # ----- Experiment paths -----
    exp_path = os.path.join("outputs", args.divergence, args.dataset, args.exp)
    if is_main:
        ensure_dir(exp_path)

    base_out = os.environ.get("FID_BASE_DIR",
                              os.environ.get("SCRATCH", args.fid_cache_dir))
    fid_dir = os.path.join(base_out, "GWF", args.divergence, args.dataset, args.exp)
    if is_main:
        ensure_dir(fid_dir)

    ckpt_dir = os.path.join(base_out, "checkpoints", "GWF",
                            args.divergence, args.dataset, args.exp)
    if is_main:
        ensure_dir(ckpt_dir)
    latest_ckpt = os.path.join(ckpt_dir, "latest.pt")

    ddp_barrier()

    # Save config
    if is_main:
        with open(os.path.join(exp_path, "config.json"), "w") as f:
            json.dump(vars(args), f, indent=2)

    log_path = os.path.join(exp_path, "log.txt")
    if is_main:
        with open(log_path, "w") as f:
            f.write("Start Training\n")

    ddp_barrier()

    # ----- Real images for FID (skip for 2D) -----
    if is_2d:
        real_img_dir = None
    else:
        real_img_dir = os.path.join(base_out, f"{args.dataset}_{args.image_size}_samples") if args.fid else None
        if is_main:
            real_img_dir = dump_real_images_if_needed(args, base_out, device)
    ddp_barrier()

    # ----- Data and noise samplers -----
    data_sampler = get_sampler(args)
    noise_sampler = get_noise(args)
    noise_sampler_old = get_noise_dataset(args)

    two_input = args.model_name in ("ncsnpp", "ncsnpp_embed")

    # Fixed noise for visualisation
    z_seed = noise_sampler.sample().to(device)
    zz_seed = torch.randn(args.batch_size, args.nz, device=device) if two_input else None

    # ----- Models -----
    netT, netD = get_model(args)
    netT, netD = netT.to(device), netD.to(device)

    kernel = get_kernel(args)
    if args.kernel == "learned_kernel":
        kernel = kernel.to(device)

    if use_ddp:
        netT = DDP(netT, device_ids=[local_rank], output_device=local_rank,
                   broadcast_buffers=False, find_unused_parameters=True)
        netD = DDP(netD, device_ids=[local_rank], output_device=local_rank,
                   broadcast_buffers=False, find_unused_parameters=True)
        if args.kernel == "learned_kernel":
            kernel = DDP(kernel, device_ids=[local_rank], output_device=local_rank,
                         broadcast_buffers=False, find_unused_parameters=True)

    if args.kernel == "learned_kernel":
        params_T = list(netT.parameters()) + list(kernel.parameters())
        params_D = list(netD.parameters()) + list(kernel.parameters())
    else:
        params_T = netT.parameters()
        params_D = netD.parameters()

    # JKO: initialise the "previous" generator
    netT_old = None
    if args.JKO:
        if two_input:
            netT_old = lambda x, z: x
        elif args.model_name in ("unet", "mlp_2d"):
            netT_old = lambda x: x
        else:
            netT_old = lambda x: noise_sampler_old.sample().to(device)

    # ----- Optimisers -----
    if args.optimizer == "adam":
        optimizerT = optim.Adam(params_T, lr=args.lr_T, betas=(args.beta1, args.beta2))
        optimizerD = optim.Adam(params_D, lr=args.lr_D, betas=(args.beta1, args.beta2))
    elif args.optimizer == "rms":
        optimizerT = optim.RMSprop(params_T, lr=args.lr_T, alpha=args.rms_alpha)
        optimizerD = optim.RMSprop(params_D, lr=args.lr_D, alpha=args.rms_alpha)
    else:
        raise ValueError(f"Unknown optimizer: {args.optimizer}")

    # EMA
    ema = EMA(optimizerT, ema_decay=args.ema_decay) if args.use_ema else None

    # ----- LR schedulers -----
    schedulerT, schedulerD = None, None
    if args.lr_scheduler:
        if args.scheduler_type == "cosine":
            total_g = max(1, ((args.JKO_steps - 1) * args.inner_iterations
                              + args.inner_iterations_first) * args.T_steps)
            total_d = max(1, ((args.JKO_steps - 1) * args.inner_iterations
                              + args.inner_iterations_first) * args.D_steps)
            schedulerT = optim.lr_scheduler.CosineAnnealingLR(
                optimizerT, T_max=total_g, eta_min=args.eta_min)
            schedulerD = optim.lr_scheduler.CosineAnnealingLR(
                optimizerD, T_max=total_d, eta_min=args.eta_min)
        else:
            raise ValueError(f"Unknown scheduler_type: {args.scheduler_type}")

    # ----- Metrics -----
    metrics = {
        "fid": {"JKO_step": [], "inner_step": [], "value": []},
        "mmd2": {"JKO_step": [], "inner_step": [], "value": []},
        "inception_score": {"JKO_step": [], "inner_step": [], "value": []},
        "kid": {"JKO_step": [], "inner_step": [], "value": []},
    }
    metrics_path = os.path.join(exp_path, "metrics.json")

    # ----- Resume from checkpoint -----
    step_generator = 0
    start_k = 0
    if args.start_from_ckpt:
        ckpt_path = latest_ckpt if args.start_from_ckpt == "latest" else args.start_from_ckpt
        start_k, step_generator, ckpt_metrics, netT_old = load_ckpt(
            ckpt_path, device,
            unwrap_model(netT), unwrap_model(netD),
            unwrap_model(kernel) if args.kernel == "learned_kernel" else None,
            netT_old, optimizerT, optimizerD, schedulerT, schedulerD, ema)
        if ckpt_metrics is not None:
            metrics = ckpt_metrics

    start_time = datetime.now()

    # ================================================================
    # Main training loop
    # ================================================================
    for k in range(start_k, args.JKO_steps):
        N = args.inner_iterations_first if k == 0 else args.inner_iterations

        for u in range(N):
            netD.train()
            netT.train()
            if args.kernel == "learned_kernel":
                kernel.train()

            # =========================
            # (1) Optimise discriminator
            # =========================
            for p in netD.parameters():
                p.requires_grad = True
            for p in netT.parameters():
                p.requires_grad = False
            if args.kernel == "learned_kernel":
                for p in kernel.parameters():
                    p.requires_grad = True

            for _ in range(args.D_steps):
                real_images = data_sampler.sample().to(device)
                if args.gp_type == "sjko" or args.divergence == "sMMD":
                    real_images.requires_grad_()

                z = noise_sampler.sample().to(device)
                with torch.no_grad():
                    if two_input:
                        zz = torch.randn(args.batch_size, args.nz, device=device)
                        fake_images = netT(z, zz)
                    else:
                        fake_images = netT(z)

                if args.gp_type == "r3gan":
                    real_images = real_images.detach().requires_grad_(True)
                    fake_images = fake_images.detach().requires_grad_(True)

                d_real = netD(real_images).view(args.batch_size, -1)
                d_fake = netD(fake_images).view(args.batch_size, -1)

                d_loss = compute_D_loss(args, z, real_images, d_real, d_fake, kernel)
                if args.lbda_gp > 0:
                    gp = gradient_penalty(
                        args, d_real, d_fake, real_images, fake_images, netD).mean()
                    d_loss = d_loss + args.lbda_gp * gp

                optimizerD.zero_grad(set_to_none=True)
                d_loss.backward()
                if args.clip_norm_grad:
                    torch.nn.utils.clip_grad_norm_(params_D, max_norm=1.0)
                optimizerD.step()

                if schedulerD is not None and args.scheduler_type == "cosine":
                    schedulerD.step()

            # =========================
            # (2) Optimise generator
            # =========================
            for p in netD.parameters():
                p.requires_grad = False
            for p in netT.parameters():
                p.requires_grad = True
            if args.kernel == "learned_kernel":
                for p in kernel.parameters():
                    p.requires_grad = True

            for _ in range(args.T_steps):
                if args.divergence in ("MMD", "iMMD", "sMMD", "chi2_tight"):
                    real_images = data_sampler.sample().to(device)
                    if args.divergence == "sMMD":
                        real_images = real_images.detach().requires_grad_(True)
                    d_real = netD(real_images).view(args.batch_size, -1)

                z = noise_sampler.sample().to(device)
                if two_input:
                    zz = torch.randn(args.batch_size, args.nz, device=device)
                    fake_images = netT(z, zz)
                else:
                    fake_images = netT(z)

                if args.JKO:
                    with torch.no_grad():
                        if two_input:
                            fake_images_old = netT_old(z.detach(), zz.detach()).detach()
                        else:
                            fake_images_old = netT_old(z.detach()).detach()

                d_fake = netD(fake_images).view(args.batch_size, -1)

                g_loss = compute_G_loss(args, z, real_images, d_real, d_fake, kernel)

                if args.JKO:
                    cost = transport_cost(
                        fake_images_old.view(args.batch_size, -1),
                        fake_images.view(args.batch_size, -1)).mean()
                    g_loss = g_loss + (1.0 / args.lambda2) * cost

                optimizerT.zero_grad(set_to_none=True)
                g_loss.backward()
                if args.clip_norm_grad:
                    torch.nn.utils.clip_grad_norm_(params_T, max_norm=1.0)
                optimizerT.step()

                if ema is not None:
                    ema.update()
                if schedulerT is not None and args.scheduler_type == "cosine":
                    schedulerT.step()

                step_generator += 1

            # ----- Logging -----
            if is_main and step_generator % args.print_every == 0:
                with open(log_path, 'a') as f:
                    f.write(f'JKO step {k} | iter {step_generator:07d} | '
                            f'G {g_loss.item():.4f} | D {d_loss.item():.4f} | '
                            f'elapsed {datetime.now() - start_time}\n')

        # =========================
        # Save samples
        # =========================
        if k % args.save_image_every == 0:
            ddp_barrier()
            if is_main:
                netT.eval()
                if ema is not None:
                    ema.swap_parameters_with_ema(store_params_in_ema=True)
                with torch.no_grad():
                    if two_input:
                        fake = netT(z_seed, zz_seed)
                    else:
                        fake = netT(z_seed)
                    if is_2d:
                        real_vis = data_sampler.sample().to(device)
                        save_2d_scatter(
                            real_vis, fake,
                            os.path.join(exp_path, f"scatter_step_{k}.png"),
                            title=f"JKO step {k}")
                    else:
                        fake = to_0_1(fake, args.images_in_0_1).cpu()
                        save_image(fake, os.path.join(exp_path, f"samples_step_{k}.png"), nrow=8)
                if ema is not None:
                    ema.restore_from_backup()
                netT.train()
            ddp_barrier()

        # =========================
        # Evaluation (FID + MMD)
        # =========================
        if k % args.evaluate_every == 0:
            ddp_barrier()
            if is_main:
                netT.eval()
                if ema is not None:
                    ema.swap_parameters_with_ema(store_params_in_ema=True)

                if args.compute_MMD:
                    with torch.no_grad():
                        if two_input:
                            fake_eval = netT(z_seed, zz_seed).view(args.batch_size, -1)
                        else:
                            fake_eval = netT(z_seed).view(args.batch_size, -1)
                        real_eval = data_sampler.sample().to(device).view(args.batch_size, -1)
                        mmd_val = MMD2(real_eval.float(), fake_eval.float(), kernel,
                                       unbiased=args.mmd_unbiased)
                    with open(log_path, "a") as f:
                        f.write(f"[EVAL] step {step_generator} | MMD^2 {mmd_val:.6f}\n")
                    append_metric(metrics, "mmd2", k, step_generator, float(mmd_val))
                    save_metrics_json(metrics, metrics_path)

                need_fake_dir = (args.fid or args.inception_score or args.kid) and not is_2d
                if need_fake_dir:
                    fake_dir = os.path.join(fid_dir, "generated_samples")
                    ensure_dir(fake_dir)
                    generate_fake_dir(args, fid_dir, fake_dir, netT, noise_sampler, device)

                if args.fid:
                    fid_value = calculate_fid_given_paths(
                        [real_img_dir, fake_dir],
                        batch_size=args.fid_batch_size, device=device, dims=2048,
                        num_workers=0)
                    with open(log_path, "a") as f:
                        f.write(f"[EVAL] JKO step {k} | step {step_generator} | "
                                f"FID {fid_value:.2f}\n")
                    append_metric(metrics, "fid", k, step_generator, fid_value)
                    save_metrics_json(metrics, metrics_path)

                if args.kid:
                    kid_mean, kid_std = compute_kid_from_dirs(
                        real_img_dir, fake_dir, device,
                        batch_size=args.fid_batch_size)
                    with open(log_path, "a") as f:
                        f.write(f"[EVAL] JKO step {k} | step {step_generator} | "
                                f"KID {kid_mean:.4f} +/- {kid_std:.4f}\n")
                    append_metric(metrics, "kid", k, step_generator, kid_mean)
                    save_metrics_json(metrics, metrics_path)

                if args.inception_score:
                    is_mean, is_std = compute_inception_score_from_dir(
                        fake_dir, device,
                        batch_size=args.fid_batch_size, splits=10)
                    with open(log_path, "a") as f:
                        f.write(f"[EVAL] JKO step {k} | step {step_generator} | "
                                f"IS {is_mean:.2f} +/- {is_std:.2f}\n")
                    append_metric(metrics, "inception_score", k, step_generator, is_mean)
                    save_metrics_json(metrics, metrics_path)

                if need_fake_dir and os.path.isdir(fake_dir) and args.delete_fake_dir:
                    shutil.rmtree(fake_dir)

                if ema is not None:
                    ema.restore_from_backup()
                netT.train()
            ddp_barrier()

        # =========================
        # JKO: update T_old
        # =========================
        if args.JKO:
            with torch.no_grad():
                netT_old = copy.deepcopy(unwrap_model(netT)).to(device)
                netT_old.requires_grad_(False)
                netT_old.eval()

        # =========================
        # Save checkpoint
        # =========================
        if args.save_ckpt:
            ddp_barrier()
            if is_main:
                save_ckpt(latest_ckpt, args,
                          unwrap_model(netT), unwrap_model(netD),
                          unwrap_model(kernel) if args.kernel == "learned_kernel" else None,
                          netT_old, optimizerT, optimizerD, schedulerT, schedulerD,
                          ema, metrics, k_next=k + 1, step_generator=step_generator)
            ddp_barrier()

    # ----- Final samples -----
    ddp_barrier()
    if is_main:
        if ema is not None:
            ema.swap_parameters_with_ema(store_params_in_ema=True)
        with torch.no_grad():
            if two_input:
                samples_final = netT(z_seed, zz_seed)
            else:
                samples_final = netT(z_seed)
            if is_2d:
                real_vis = data_sampler.sample().to(device)
                save_2d_scatter(
                    real_vis, samples_final,
                    os.path.join(exp_path, "final_scatter.png"),
                    title="Final")
            else:
                images = to_0_1(samples_final, args.images_in_0_1).cpu()
                save_image(images, os.path.join(exp_path, 'final_samples.png'), nrow=8)
        if ema is not None:
            ema.restore_from_backup()
    ddp_barrier()

    # Cleanup
    if is_main and not is_2d:
        fake_dir = os.path.join(fid_dir, "generated_samples")
        if (args.fid or args.inception_score) and args.delete_fake_dir and os.path.isdir(fake_dir):
            shutil.rmtree(fake_dir)

    if is_main:
        print(f"Training complete. Results saved to {exp_path}")
    if use_ddp:
        dist.destroy_process_group()


# ============================================================
# CLI
# ============================================================

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Generative Wasserstein Flows (GWF)')

    # --- Experiment ---
    parser.add_argument("--exp", default="test", help="Experiment name")
    parser.add_argument("--dataset", default="cifar10",
                        choices=["mnist", "mnist28", "cifar10", "celeba", "celeba_hq",
                                 "circles", "three_rings", "eight_gaussians", "moons"])
    parser.add_argument("--seed", type=int, default=1024)
    parser.add_argument("--image_size", type=int, default=32)
    parser.add_argument("--num_channels", type=int, default=3)

    # --- Paths ---
    parser.add_argument("--data_root", type=str, default="../data",
                        help="Root directory for datasets (downloaded here if missing)")
    parser.add_argument("--fid_cache_dir", type=str, default="./fid_cache",
                        help="Directory for FID/IS generated images and checkpoints. "
                             "Overridden by FID_BASE_DIR or SCRATCH env vars if set.")

    # --- JKO scheme ---
    parser.add_argument("--JKO", action=argparse.BooleanOptionalAction, default=False,
                        help="Enable JKO proximal scheme (GWF). Without this, trains a standard GAN.")
    parser.add_argument("--lambda2", type=float, default=1.0,
                        help="JKO step size (1/lambda2 scales the transport cost)")
    parser.add_argument("--JKO_steps", type=int, default=10,
                        help="Number of outer JKO steps")

    # --- Divergence ---
    parser.add_argument("--divergence", default="MMD",
                        choices=["MMD", "ckMMD", "sMMD", "iMMD", "Wasserstein-1",
                                 "KL", "KL_centered", "Shannon", "chi2",
                                 "KL_DV", "chi2_tight"],
                        help="Divergence to minimise")
    parser.add_argument("--symmetric", action=argparse.BooleanOptionalAction,
                        default=False,
                        help="Minimise D_f(nu || mu) instead of D_f(mu || nu). "
                             "Coincides with f-GANs for f-divergences. "
                             "No effect for symmetric divergences (MMD, W-1, Shannon).")
    parser.add_argument("--gp_type", default="interpolated",
                        choices=["sjko", "interpolated", "r3gan", "WGAN"],
                        help="Gradient penalty type")
    parser.add_argument("--lbda_gp", type=float, default=10.0,
                        help="Gradient penalty weight (0 to disable)")
    parser.add_argument("--images_in_0_1", action=argparse.BooleanOptionalAction,
                        default=False,
                        help="If True, images in [0,1]; if False (default), in [-1,1]")

    # --- Training ---
    parser.add_argument("--D_steps", type=int, default=1, help="Critic steps per iteration")
    parser.add_argument("--T_steps", type=int, default=1, help="Generator steps per iteration")
    parser.add_argument("--inner_iterations_first", type=int, default=2000,
                        help="Inner iterations for JKO step 0")
    parser.add_argument("--inner_iterations", type=int, default=1000,
                        help="Inner iterations for JKO steps >= 1")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr_D", type=float, default=1e-4)
    parser.add_argument("--lr_T", type=float, default=2e-4)
    parser.add_argument("--optimizer", default="adam", choices=["adam", "rms"])
    parser.add_argument("--rms_alpha", type=float, default=0.99)
    parser.add_argument("--beta1", type=float, default=0.5)
    parser.add_argument("--beta2", type=float, default=0.9)
    parser.add_argument("--type_noise", default="unif", choices=["normal", "unif"])

    # --- EMA ---
    parser.add_argument("--use_ema", action="store_true", default=False)
    parser.add_argument("--ema_decay", type=float, default=0.9999)

    # --- LR scheduler ---
    parser.add_argument("--lr_scheduler", action="store_true", default=False)
    parser.add_argument("--scheduler_type", default="cosine", choices=["cosine"])
    parser.add_argument("--eta_min", type=float, default=5e-5)

    # --- Evaluation ---
    parser.add_argument("--print_every", type=int, default=1000)
    parser.add_argument("--evaluate_every", type=int, default=10)
    parser.add_argument("--save_image_every", type=int, default=10)
    parser.add_argument("--fid", action="store_true", default=False,
                        help="Compute FID during evaluation")
    parser.add_argument("--kid", action="store_true", default=False,
                        help="Compute KID (Kernel Inception Distance) during evaluation. "
                             "Reuses the same InceptionV3 weights as FID (offline-safe).")
    parser.add_argument("--inception_score", action="store_true", default=False,
                        help="Compute Inception Score during evaluation. "
                             "Set INCEPTION_WEIGHTS_PATH env var for offline usage.")
    parser.add_argument("--fid_batch_size", type=int, default=64)
    parser.add_argument("--delete_fake_dir", action=argparse.BooleanOptionalAction,
                        default=True)
    parser.add_argument("--compute_MMD", action="store_true", default=False)

    # --- Architecture ---
    parser.add_argument("--model_name", default="resnet_MMDGAN",
                        choices=["resnet_MMDGAN", "ncsnpp", "ncsnpp_embed", "otm", "unet",
                                 "mlp_2d"])
    parser.add_argument("--embed_dim", type=int, default=128,
                        help="Discriminator output dimension (use 1 for scalar divergences)")
    parser.add_argument("--nz", type=int, default=128, help="Generator noise dimension")
    parser.add_argument("--clip_norm_grad", action=argparse.BooleanOptionalAction,
                        default=False)

    # --- MMD parameters ---
    parser.add_argument("--mmd_unbiased", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--use_sn", action=argparse.BooleanOptionalAction, default=False,
                        help="Spectral normalisation in discriminator")
    parser.add_argument("--kernel", default="learned_kernel",
                        choices=["learned_kernel", "linear", "rbf", "riesz",
                                 "riesz_psd", "imq", "rq", "mix_rq"])
    parser.add_argument("--scaling_variant", default="grad",
                        choices=["grad", "value_and_grad"])
    parser.add_argument("--scaling_coeff", type=float, default=10.0)
    parser.add_argument("--sigma_kernel", type=float, default=10.0)
    parser.add_argument("--n_hutch", type=int, default=1,
                        help="Hutchinson projections for sMMD scaling")

    # --- Architecture-specific ---
    parser.add_argument("--ngf", type=int, default=128,
                        help="Generator feature maps (used by ncsnpp, otm)")
    parser.add_argument("--ndf", type=int, default=64,
                        help="Discriminator feature maps (used by otm, unet)")

    # --- Checkpointing ---
    parser.add_argument("--save_ckpt", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--start_from_ckpt", type=str, default="",
                        help="Path to checkpoint or 'latest'. Empty = start fresh.")
    parser.add_argument("--ddp", action=argparse.BooleanOptionalAction, default=False,
                        help="Enable DistributedDataParallel when launched with torchrun/srun.")

    args = parser.parse_args()
    train(args)
