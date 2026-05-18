"""Evaluation utilities: FID, Inception Score, KID, metric logging, checkpointing."""

import os
import math
import glob
import json

import numpy as np
import torch
import torch.nn.functional as F
import torchvision
from torchvision.utils import save_image

from dataset import get_sampler
from utils import ensure_dir, to_0_1


# ============================================================
# Real / fake image directories for FID
# ============================================================

@torch.no_grad()
def dump_real_images_if_needed(args, base_out, device):
    """Save all real training images as PNGs for FID computation.

    Returns the path to the directory, or None if FID is disabled.
    """
    if not args.fid:
        return None

    real_img_dir = os.path.join(base_out, f"{args.dataset}_{args.image_size}_samples")
    ensure_dir(real_img_dir)

    train_loader = get_sampler(args).loader
    nb_samples = len(train_loader.dataset)
    existing = (len(glob.glob(os.path.join(real_img_dir, "*.png")))
                + len(glob.glob(os.path.join(real_img_dir, "*.jpg"))))

    if existing >= nb_samples:
        return real_img_dir

    idx = existing
    for x, _ in train_loader:
        x = to_0_1(x.to(device), args.images_in_0_1).cpu()
        for i in range(x.size(0)):
            save_image(x[i], os.path.join(real_img_dir, f"{idx:05d}.png"))
            idx += 1

    return real_img_dir


@torch.no_grad()
def generate_fake_dir(args, fid_dir, fake_dir, netT, noise_sampler, device):
    """Generate 50k fake samples and save them as PNGs for FID."""
    ensure_dir(fake_dir)

    # Remove old images
    for f in glob.glob(os.path.join(fake_dir, "*.png")):
        os.remove(f)

    two_input = args.model_name in ("ncsnpp", "ncsnpp_embed")
    num_samples = 50_000
    iters_needed = num_samples // args.batch_size

    netT.eval()
    for i in range(iters_needed):
        noise = noise_sampler.sample().to(device)
        if two_input:
            zz = torch.randn(args.batch_size, args.nz, device=device)
            fake = netT(noise, zz)
        else:
            fake = netT(noise)
        fake = to_0_1(fake.float(), args.images_in_0_1)
        for j, img in enumerate(fake):
            index = i * args.batch_size + j
            save_image(img, os.path.join(fake_dir, f"{index:05d}.png"))
        if i % 50 == 0:
            print(f"  [FID] generating batch {i}/{iters_needed}", end="\r")
    print()
    netT.train()


# ============================================================
# Inception Score
# ============================================================

def _hub_dir():
    """Return the torch hub checkpoints directory."""
    torch_home = os.environ.get(
        "TORCH_HOME",
        os.path.join(os.path.expanduser("~"), ".cache", "torch"),
    )
    return os.path.join(torch_home, "hub", "checkpoints")


def load_inception_v3(device):
    """Load an Inception-v3 model for computing the Inception Score.

    Weight loading priority (offline-first, so it works on Jean Zay compute
    nodes without network access):

    1. ``$TORCH_HOME/hub/checkpoints/inception_v3_google-0cc3c7bd.pth``
       — torchvision's standard cache path.  Set ``TORCH_HOME`` to point
       to your work directory and pre-download this file on a login node.
    2. ``INCEPTION_WEIGHTS_PATH`` env variable — treated as a torchvision
       state-dict (same architecture, 1000 classes).
    3. ``pytorch_fid.inception.fid_inception_v3()`` — loads the same
       ``pt_inception-2015-12-05-6726825d.pth`` already cached by FID/KID.
       Outputs 1008 classes instead of 1000; IS values are comparable.
       **This is the zero-extra-download fallback for offline clusters.**
    4. Online torchvision download (last resort).

    Returns:
        An Inception-v3 model in eval mode on ``device``.
    """
    from torchvision.models import inception_v3

    def _load_tv_state_dict(path):
        """Load a torchvision inception_v3 state dict (1000 classes)."""
        sd = torch.load(path, map_location="cpu")
        if isinstance(sd, dict) and "state_dict" in sd:
            sd = sd["state_dict"]
        model = inception_v3(weights=None, aux_logits=True)
        model.load_state_dict(sd, strict=True)
        model.aux_logits = False
        model.AuxLogits = None
        return model.to(device).eval()

    # 1. Torchvision standard hub location
    tv_path = os.path.join(_hub_dir(), "inception_v3_google-0cc3c7bd.pth")
    if os.path.isfile(tv_path):
        print(f"[IS] Loading torchvision Inception-v3 from {tv_path}")
        return _load_tv_state_dict(tv_path)

    # 2. Explicit INCEPTION_WEIGHTS_PATH env var
    env_path = os.environ.get("INCEPTION_WEIGHTS_PATH", "")
    if env_path and os.path.isfile(env_path):
        print(f"[IS] Trying INCEPTION_WEIGHTS_PATH: {env_path}")
        try:
            return _load_tv_state_dict(env_path)
        except Exception as e:
            print(f"[IS]   Could not load as torchvision state dict ({e}); "
                  "falling back to pytorch-fid model.")

    # 3. pytorch-fid's fid_inception_v3 (pt_inception weights, offline-safe)
    #    This model uses the same .pth file that FID/KID already cached, so no
    #    download is needed if FID has ever been run with the same TORCH_HOME.
    #
    #    IMPORTANT: fid_inception_v3 weights were trained with images in [-1, 1]
    #    (the pytorch-fid InceptionV3 wrapper applies `2*x - 1` before forwarding).
    #    compute_inception_score_from_dir passes [0, 1] images, so we wrap the
    #    model to apply that same normalisation, otherwise all logits collapse to
    #    a uniform distribution and IS = 1.00 exactly.
    try:
        from pytorch_fid.inception import fid_inception_v3

        class _FidInceptionForIS(torch.nn.Module):
            """fid_inception_v3 with [0,1]→[-1,1] input normalisation."""
            def __init__(self, base):
                super().__init__()
                self.base = base
            def forward(self, x):
                return self.base(2.0 * x - 1.0)

        print("[IS] Loading pytorch-fid fid_inception_v3 (1008 classes, offline).")
        model = _FidInceptionForIS(fid_inception_v3())
        return model.to(device).eval()
    except Exception as e:
        print(f"[IS]   fid_inception_v3 fallback failed: {e}")

    # 4. Online download (last resort — will fail on offline nodes)
    print("[IS] Downloading Inception-v3 weights from torchvision (requires network).")
    from torchvision.models import Inception_V3_Weights
    model = inception_v3(weights=Inception_V3_Weights.DEFAULT, aux_logits=True)
    model.aux_logits = False
    model.AuxLogits = None
    return model.to(device).eval()


@torch.no_grad()
def compute_inception_score_from_dir(fake_dir, device, batch_size=64,
                                     splits=10, num_workers=2):
    """Compute the Inception Score (IS) from a directory of generated images.

    IS = exp( E_x[ KL( p(y|x) || p(y) ) ] ), averaged over ``splits``.

    Uses a streaming approach: probabilities are accumulated per-split
    without storing all of them in memory.

    Args:
        fake_dir:    Directory containing generated PNG/JPG images.
        device:      Torch device.
        batch_size:  Batch size for inference.
        splits:      Number of splits for mean/std computation.
        num_workers: DataLoader workers.

    Returns:
        (is_mean, is_std): Inception Score mean and standard deviation.
    """
    # Gather image files
    exts = ("*.jpg", "*.jpeg", "*.png", "*.bmp")
    files = []
    for ext in exts:
        files.extend(glob.glob(os.path.join(fake_dir, ext)))
    files = sorted(files)
    N = len(files)
    assert N > 0, f"No images found in {fake_dir}"

    batch_size = max(1, min(batch_size, N))
    splits = max(1, min(splits, N))

    # Load Inception model
    model = load_inception_v3(device)

    # Simple dataset that reads image files
    class _FileDataset(torch.utils.data.Dataset):
        def __init__(self, files):
            self.files = files
        def __len__(self):
            return len(self.files)
        def __getitem__(self, i):
            x = torchvision.io.read_image(self.files[i])  # [C, H, W] uint8
            x = x.float().div_(255.0)                      # [0, 1]
            if x.size(0) == 1:
                x = x.repeat(3, 1, 1)                      # grayscale -> 3ch
            return x

    dl = torch.utils.data.DataLoader(
        _FileDataset(files), batch_size=batch_size, shuffle=False,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
    )

    # Prepare split boundaries
    base, rem = divmod(N, splits)
    split_sizes = [base + (1 if s < rem else 0) for s in range(splits)]

    # Streaming accumulators per split
    sum_probs = None   # will be [splits] list of [K] tensors
    sum_plogp = None   # will be [splits] list of floats
    count_split = [0] * splits
    cur_split = 0
    remaining = split_sizes[0]

    for xb in dl:
        xb = xb.to(device, non_blocking=True)
        xb = F.interpolate(xb, size=(299, 299), mode="bilinear", align_corners=False)
        logits = model(xb)
        if isinstance(logits, (tuple, list)):
            logits = logits[0]
        probs = F.softmax(logits, dim=1)  # [B, K]
        B, K = probs.shape

        # Lazy init (K unknown until first batch)
        if sum_probs is None:
            sum_probs = [torch.zeros(K, device=device) for _ in range(splits)]
            sum_plogp = [0.0] * splits

        # Distribute batch across splits
        offset = 0
        while offset < B and cur_split < splits:
            take = min(remaining, B - offset)
            p = probs[offset:offset + take]
            sum_probs[cur_split] += p.sum(dim=0)
            sum_plogp[cur_split] += float((p * p.clamp_min(1e-12).log()).sum().item())
            count_split[cur_split] += take
            offset += take
            remaining -= take
            if remaining == 0:
                cur_split += 1
                if cur_split < splits:
                    remaining = split_sizes[cur_split]

    # Compute IS per split
    scores = []
    for s in range(splits):
        n = count_split[s]
        if n == 0:
            continue
        py = (sum_probs[s] / n).clamp_min(1e-12)       # marginal p(y)
        term1 = sum_plogp[s] / n                         # E[sum p log p]
        term2 = float((py * py.log()).sum().item())       # sum p_bar log p_bar
        scores.append(math.exp(term1 - term2))

    if not scores:
        return float("nan"), float("nan")

    scores = np.array(scores)
    return float(scores.mean()), float(scores.std())


# ============================================================
# Kernel Inception Distance (KID)
# ============================================================

def _polynomial_mmd(X: torch.Tensor, Y: torch.Tensor,
                    degree: int = 3, gamma: float = None,
                    coef0: float = 1.0) -> float:
    """Unbiased MMD² with a polynomial kernel between feature matrices.

    k(x, y) = (gamma * x·y + coef0)^degree

    Args:
        X: [n, d] real-valued feature matrix.
        Y: [m, d] real-valued feature matrix.
        degree: Polynomial degree (default 3, as in Binkowski et al. 2018).
        gamma:  Kernel bandwidth; defaults to 1/d.
        coef0:  Bias term (default 1.0).

    Returns:
        Scalar unbiased MMD² estimate.
    """
    n, d = X.shape
    m = Y.shape[0]
    if gamma is None:
        gamma = 1.0 / d

    K_XX = (gamma * X.mm(X.t()) + coef0).pow(degree)   # [n, n]
    K_YY = (gamma * Y.mm(Y.t()) + coef0).pow(degree)   # [m, m]
    K_XY = (gamma * X.mm(Y.t()) + coef0).pow(degree)   # [n, m]

    # Unbiased estimator: exclude diagonal for within-set terms
    mmd2 = (K_XX.sum() - K_XX.trace()) / (n * (n - 1))
    mmd2 += (K_YY.sum() - K_YY.trace()) / (m * (m - 1))
    mmd2 -= 2.0 * K_XY.mean()
    return float(mmd2.item())


@torch.no_grad()
def _extract_inception_features(image_dir: str, device: torch.device,
                                 batch_size: int = 64, dims: int = 2048,
                                 num_workers: int = 0) -> np.ndarray:
    """Extract InceptionV3 pool3 features from a directory of images.

    Uses the same ``pytorch_fid.inception.InceptionV3`` model as
    ``calculate_fid_given_paths``, so the weights are already cached in
    ``TORCH_HOME`` after a first FID run (fully offline-compatible on Jean Zay).

    Args:
        image_dir:   Directory containing PNG/JPG images.
        device:      Torch device.
        batch_size:  Inference batch size.
        dims:        Feature dimensionality (2048 = pool3).
        num_workers: DataLoader workers.

    Returns:
        Float32 numpy array of shape [N, dims].
    """
    from pytorch_fid.inception import InceptionV3
    from pytorch_fid.fid_score import get_activations

    block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[dims]
    model = InceptionV3([block_idx]).to(device).eval()

    # get_activations expects a list of file paths and the model
    exts = ("*.jpg", "*.jpeg", "*.png", "*.bmp")
    files = []
    for ext in exts:
        files.extend(glob.glob(os.path.join(image_dir, ext)))
    files = sorted(files)
    assert len(files) > 0, f"No images found in {image_dir}"

    features = get_activations(files, model, batch_size=batch_size,
                               dims=dims, device=device, num_workers=num_workers)
    return features


def compute_kid_from_dirs(real_dir: str, fake_dir: str,
                          device: torch.device,
                          batch_size: int = 64,
                          dims: int = 2048,
                          n_subsets: int = 100,
                          subset_size: int = 1000,
                          num_workers: int = 0) -> tuple:
    """Compute Kernel Inception Distance (KID) between two image directories.

    KID = MMD²(real_features, fake_features) with a degree-3 polynomial kernel
    on InceptionV3 pool3 features (Binkowski et al., 2018).

    Variance is estimated by repeating the MMD² computation on ``n_subsets``
    random subsets of size ``subset_size``, following the reference
    implementation.

    Offline-safe: uses the same InceptionV3 weights as ``calculate_fid_given_paths``
    which are cached in ``$TORCH_HOME/hub/checkpoints/`` after the first FID run.

    Args:
        real_dir:    Directory with real PNG/JPG images.
        fake_dir:    Directory with generated PNG/JPG images.
        device:      Torch device.
        batch_size:  Batch size for Inception inference.
        dims:        Inception feature dimension (default 2048).
        n_subsets:   Number of random subsets for variance estimation.
        subset_size: Number of samples per subset.
        num_workers: DataLoader workers for feature extraction.

    Returns:
        (kid_mean, kid_std): KID × 100 (×100 is conventional for readability).
    """
    print("[KID] Extracting features from real images...")
    real_feats = _extract_inception_features(real_dir, device, batch_size,
                                             dims, num_workers)
    print("[KID] Extracting features from fake images...")
    fake_feats = _extract_inception_features(fake_dir, device, batch_size,
                                             dims, num_workers)

    real_t = torch.from_numpy(real_feats).to(device, dtype=torch.float32)
    fake_t = torch.from_numpy(fake_feats).to(device, dtype=torch.float32)

    n_real = real_t.shape[0]
    n_fake = fake_t.shape[0]
    subset_size = min(subset_size, n_real, n_fake)

    scores = []
    for _ in range(n_subsets):
        idx_r = torch.randperm(n_real, device=device)[:subset_size]
        idx_f = torch.randperm(n_fake, device=device)[:subset_size]
        mmd2 = _polynomial_mmd(real_t[idx_r], fake_t[idx_f])
        scores.append(mmd2)

    scores = np.array(scores)
    # Multiply by 100 (standard convention for KID reporting)
    return float(scores.mean() * 100), float(scores.std() * 100)


# ============================================================
# Metric logging
# ============================================================

def append_metric(metrics: dict, key: str, k: int, step: int, value: float):
    """Append a metric value to the metrics dictionary."""
    metrics[key]["JKO_step"].append(int(k))
    metrics[key]["inner_step"].append(int(step))
    metrics[key]["value"].append(float(value))


def save_metrics_json(metrics: dict, path: str):
    """Atomically write metrics to a JSON file."""
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(metrics, f, indent=2)
    os.replace(tmp, path)


# ============================================================
# Checkpointing
# ============================================================

def save_ckpt(ckpt_path, args, netT, netD, kernel, netT_old,
              optimizerT, optimizerD, schedulerT, schedulerD,
              ema, metrics, k_next, step_generator):
    """Save a training checkpoint."""
    payload = {
        "args": vars(args),
        "k_next": int(k_next),
        "step_generator": int(step_generator),
        "netT": netT.state_dict(),
        "netD": netD.state_dict(),
        "kernel": (kernel.state_dict()
                   if kernel is not None and hasattr(kernel, "state_dict") else None),
        "netT_old": (netT_old.state_dict()
                     if netT_old is not None and hasattr(netT_old, "state_dict") else None),
        "optimizerT": optimizerT.state_dict(),
        "optimizerD": optimizerD.state_dict(),
        "schedulerT": schedulerT.state_dict() if schedulerT is not None else None,
        "schedulerD": schedulerD.state_dict() if schedulerD is not None else None,
        "ema": ema.state_dict() if ema is not None else None,
        "metrics": metrics,
    }
    ensure_dir(os.path.dirname(ckpt_path))
    torch.save(payload, ckpt_path)


def load_ckpt(ckpt_path, device, netT, netD, kernel, netT_old,
              optimizerT, optimizerD, schedulerT, schedulerD, ema):
    """Load a training checkpoint. Returns (k_next, step_generator, metrics, netT_old)."""
    ckpt = torch.load(ckpt_path, map_location=device)

    netT.load_state_dict(ckpt["netT"], strict=True)
    netD.load_state_dict(ckpt["netD"], strict=True)

    if ckpt.get("kernel") is not None and kernel is not None:
        kernel.load_state_dict(ckpt["kernel"], strict=True)

    optimizerT.load_state_dict(ckpt["optimizerT"])
    optimizerD.load_state_dict(ckpt["optimizerD"])

    if schedulerT is not None and ckpt.get("schedulerT") is not None:
        schedulerT.load_state_dict(ckpt["schedulerT"])
    if schedulerD is not None and ckpt.get("schedulerD") is not None:
        schedulerD.load_state_dict(ckpt["schedulerD"])

    if ema is not None and ckpt.get("ema") is not None:
        ema.load_state_dict(ckpt["ema"])

    # Restore netT_old
    if ckpt.get("netT_old") is not None and hasattr(netT_old, "load_state_dict"):
        netT_old.load_state_dict(ckpt["netT_old"], strict=True)
        netT_old.requires_grad_(False)
        netT_old.eval()

    k_next = int(ckpt.get("k_next", 0))
    step_generator = int(ckpt.get("step_generator", 0))
    metrics = ckpt.get("metrics", None)

    return k_next, step_generator, metrics, netT_old
