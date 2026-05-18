# Generative Wasserstein Flows (GWF)

Official implementation of:

> **A Unifying View of Variational Generative Wasserstein Flows**  
> Paul Caucheteux, Clément Bonet, Anna Korba.  
> *International Conference on Machine Learning (ICML), 2026.*

## Overview

GWF implements the JKO (Jordan-Kinderlehrer-Otto) proximal scheme for generative modelling.
At each outer step the generator solves the proximal problem:

$$T_{k+1} = \arg\min_T \{ D(T_\sharp\mu_0 \| \nu) + \frac{1}{2\tau} ||T - T_k||^2_{L^2(\mu_0)} \}$$

It supports **f-divergences** (KL, chi², Jensen-Shannon), **IPMs** (Wasserstein-1), and **squared MMD** variants (MMD, ckMMD, sMMD, iMMD).
The Donsker-Varadhan (DV) formulation of the KL divergence (`KL_DV`) is also supported and generally yields improved results (see Section 5.1 of the paper).

## Installation

```bash
pip install -r requirements.txt
```

Datasets (MNIST, CIFAR-10) are downloaded automatically on first run.
For CelebA, place the images under `../data/celeba/img_align_celeba/` (or set `--data_root`).

## Evaluation

FID is computed with the [pytorch-fid](https://github.com/mseitzer/pytorch-fid) package.
Enable it with `--fid`; generated and real images are saved to a cache directory
(`--fid_cache_dir`, or `$SCRATCH` / `$FID_BASE_DIR` on clusters).
KID (`--kid`) and Inception Score (`--inception_score`) are also available.

## Repository structure

```
GWF/
├── train.py              # Main training script (JKO + all divergences)
├── dataset.py            # Data samplers (CIFAR-10, MNIST, CelebA, 2D toys)
├── losses.py             # Divergence losses (D/G, forward & symmetric), gradient penalties
├── kernels.py            # Kernel functions for MMD (fixed + learned)
├── mmd.py                # MMD² computation and scaling
├── eval.py               # FID, KID, Inception Score, checkpointing
├── utils.py              # EMA, Jacobian norms, misc helpers
├── networks/             # All generator/discriminator architectures
│   ├── __init__.py       # Model factory (get_model) + ResNet, MLP-2D
│   ├── unet.py           # UNet generator
│   ├── ncsnpp/           # NCSN++ generator + discriminators (Large-Net)
│   └── modelOTM/         # OTM generator + discriminators (Small-Net)
└── requirements.txt
```

## Example runs

> **Note on `--lambda2`:** `--lambda2` corresponds to 2τ with τ the step size used in
> the paper (`1/lambda2` scales the transport cost).

### 2D toy datasets (quick sanity check)

```bash
python train.py \
    --dataset eight_gaussians --divergence KL_DV --JKO \
    --model_name mlp_2d --nz 2 --embed_dim 1 \
    --JKO_steps 5 --inner_iterations_first 500 --inner_iterations 500 \
    --batch_size 256 --gp_type sjko --lbda_gp 0.2 --print_every 100 \
    --evaluate_every 1 --save_image_every 1 --compute_MMD \
    --exp eight_gaussians_kldv
```

### CIFAR-10 — Large-Net + KL_DV + JKO (Table 3, Figure 1, FID ≈ 9)

```bash
python train.py \
    --dataset cifar10 --divergence KL_DV --JKO --lambda2 0.2 \
    --model_name ncsnpp --nz 128 --batch_size 256 \
    --JKO_steps 50 --inner_iterations_first 2000 --inner_iterations 2000 \
    --lr_scheduler --scheduler_type cosine --eta_min 5e-5 \
    --gp_type sjko --lbda_gp 0.2 --use_ema --ema_decay 0.9999 \
    --fid --evaluate_every 10 --save_image_every 10 --save_ckpt \
    --exp cifar10_ncsnpp_kldv_jko
```

### CIFAR-10 — Small-Net (OTM) + MMD + JKO (Table 1, FID ≈ 16)

```bash
python train.py \
    --dataset cifar10 --divergence MMD --JKO --lambda2 1 \
    --model_name otm --nz 192 --ngf 256 --embed_dim 128 --batch_size 256 \
    --JKO_steps 100 --inner_iterations_first 2000 --inner_iterations 1000 \
    --gp_type interpolated --lbda_gp 10 --mmd_unbiased \
    --use_ema --ema_decay 0.9999 \
    --fid --evaluate_every 10 --save_image_every 10 --save_ckpt \
    --exp cifar10_otm_mmd_jko
```

### CIFAR-10 — Small-Net (OTM) + KL_DV + JKO (Table 4 / Figure 2)

```bash
python train.py \
    --dataset cifar10 --divergence KL_DV --JKO --lambda2 1 \
    --model_name otm --nz 192 --ngf 256 --embed_dim 1 --batch_size 256 \
    --JKO_steps 100 --inner_iterations_first 2000 --inner_iterations 1000 \
    --gp_type sjko --lbda_gp 0.2 --use_ema --ema_decay 0.9999 \
    --fid --evaluate_every 10 --save_image_every 10 --save_ckpt \
    --exp cifar10_otm_kldv_jko
```

## Key command-line flags

| Flag | Description |
|---|---|
| `--JKO` / `--no-JKO` | Enable/disable the JKO proximal scheme |
| `--lambda2` | JKO step size (`1/lambda2` scales the transport cost) |
| `--divergence` | Divergence to minimise (`KL_DV`, `KL`, `MMD`, `Shannon`, `chi2`, `Wasserstein-1`, ...) |
| `--symmetric` | Use D_f(ν \|\| μ) instead of D_f(μ \|\| ν) |
| `--model_name` | Generator architecture (`ncsnpp`, `otm`, `unet`, `resnet_MMDGAN`, `mlp_2d`) |
| `--fid` / `--kid` / `--inception_score` | Enable evaluation metrics (FID uses [pytorch-fid](https://github.com/mseitzer/pytorch-fid)) |
| `--use_ema` | Exponential moving average of generator weights |
| `--ddp` | Multi-GPU training via DistributedDataParallel |
| `--save_ckpt` | Save checkpoints for resuming |

Run `python train.py --help` for the full list.

## Citation

```bibtex
@inproceedings{caucheteux2026gwf,
    title={A Unifying View of Variational Generative Wasserstein Flows},
    author={Caucheteux, Paul and Bonet, Cl{\'e}ment and Korba, Anna},
    booktitle={International Conference on Machine Learning},
    year={2026}
}
```
