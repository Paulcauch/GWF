"""Data samplers and noise samplers for GWF training."""

import math
import os

import numpy as np
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from PIL import Image


# ============================================================
# 2D datasets
# ============================================================

_2D_DATASETS = {"circles", "three_rings", "eight_gaussians", "moons"}


def is_2d_dataset(name: str) -> bool:
    """Return True if ``name`` is a 2D toy dataset."""
    return name in _2D_DATASETS


class Circles:
    """Concentric noisy circles in 2D."""
    def __init__(self, batch_size, centers=None, radius=None, sigmas=None):
        if centers is None:
            centers = [[0, 0], [0, 0]]
        if radius is None:
            radius = [4, 8]
        if sigmas is None:
            sigmas = [0.2, 0.2]
        assert len(centers) == len(radius) == len(sigmas)
        assert batch_size % len(centers) == 0

        self.batch_size = batch_size
        ind = batch_size // len(centers)
        self.centers = torch.tensor(centers * ind, dtype=torch.float32)
        self.radius = torch.tensor(radius * ind, dtype=torch.float32)[:, None]
        self.sigmas = torch.tensor(sigmas * ind, dtype=torch.float32)[:, None]

    def sample(self):
        noise = torch.randn(self.batch_size, 2)
        z = torch.randn(self.batch_size, 2)
        z = z / z.norm(dim=1, keepdim=True)
        return self.centers + self.radius * z + self.sigmas * noise


class ThreeRings:
    """Three vertically stacked rings in 2D."""
    def __init__(self, batch_size):
        self.batch_size = batch_size
        r, delta = 0.3, 0.5
        N = 1000
        ring0 = np.c_[r * np.cos(np.linspace(0, 2 * np.pi, N, endpoint=False)),
                       r * np.sin(np.linspace(0, 2 * np.pi, N, endpoint=False))]
        offset = np.array([0, (2 + delta) * r])
        ring1 = ring0 - offset
        ring2 = ring0 - 2 * offset
        self.all_points = torch.from_numpy(
            np.vstack([ring0, ring1, ring2])).float()

    def sample(self):
        idx = torch.randint(0, len(self.all_points), (self.batch_size,))
        return self.all_points[idx]


class EightGaussians:
    """Mixture of 8 isotropic Gaussians arranged in a circle."""
    def __init__(self, batch_size, scale=1.0, var=0.01):
        self.batch_size = batch_size
        self.var = var
        centers = [(1, 0), (-1, 0), (0, 1), (0, -1),
                   (1.0 / math.sqrt(2), 1.0 / math.sqrt(2)),
                   (1.0 / math.sqrt(2), -1.0 / math.sqrt(2)),
                   (-1.0 / math.sqrt(2), 1.0 / math.sqrt(2)),
                   (-1.0 / math.sqrt(2), -1.0 / math.sqrt(2))]
        self.centers = torch.tensor(centers) * scale

    def sample(self):
        idx = torch.randint(0, 8, (self.batch_size,))
        noise = math.sqrt(self.var) * torch.randn(self.batch_size, 2)
        return self.centers[idx] + noise


class Moons:
    """Two interleaving half-circles (sklearn-style), rescaled to ~[-4, 4]."""
    def __init__(self, batch_size, noise=0.2):
        self.batch_size = batch_size
        self.noise = noise

    def sample(self):
        n = self.batch_size
        n1 = n // 2
        n2 = n - n1
        theta1 = torch.rand(n1) * math.pi
        x1 = torch.stack([torch.cos(theta1), torch.sin(theta1)], dim=1)
        theta2 = torch.rand(n2) * math.pi
        x2 = torch.stack([1 - torch.cos(theta2), 1 - torch.sin(theta2) - 0.5], dim=1)
        data = torch.cat([x1, x2], dim=0)
        data = data + self.noise * torch.randn_like(data)
        perm = torch.randperm(n)
        return data[perm] * 3 - 1


# ============================================================
# Image dataset samplers
# ============================================================

class CIFAR10Sampler:
    def __init__(self, batch_size, normalize=True, train=True, shuffle=True,
                 data_root='../data/cifar10'):
        self.batch_size = batch_size
        if normalize:
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ])
        else:
            transform = transforms.Compose([transforms.ToTensor()])
        self.dataset = datasets.CIFAR10(
            root=data_root, train=train, download=True, transform=transform)
        self.loader = DataLoader(
            self.dataset, batch_size=batch_size, shuffle=shuffle,
            drop_last=True, num_workers=2, pin_memory=True)
        self.iterator = iter(self.loader)

    def sample(self):
        try:
            data, _ = next(self.iterator)
        except StopIteration:
            self.iterator = iter(self.loader)
            data, _ = next(self.iterator)
        return data


class MNISTSampler:
    def __init__(self, batch_size, normalize=True, train=True, shuffle=True,
                 data_root='../data', resize_32=True):
        self.batch_size = batch_size
        tfms = []
        if resize_32:
            tfms.append(transforms.Resize(32))
        tfms.append(transforms.ToTensor())
        if normalize:
            tfms.append(transforms.Normalize((0.5,), (0.5,)))
        self.dataset = datasets.MNIST(
            root=data_root, train=train, download=True,
            transform=transforms.Compose(tfms))
        self.loader = DataLoader(
            self.dataset, batch_size=batch_size, shuffle=shuffle,
            drop_last=True, num_workers=2, pin_memory=True)
        self.iterator = iter(self.loader)

    def sample(self):
        try:
            data, _ = next(self.iterator)
        except StopIteration:
            self.iterator = iter(self.loader)
            data, _ = next(self.iterator)
        return data


class FlatImageDataset(torch.utils.data.Dataset):
    """Dataset for folders that contain images directly (no class subfolders)."""
    IMG_EXTS = (".jpg", ".jpeg", ".png", ".ppm", ".bmp", ".pgm", ".tif", ".tiff", ".webp")

    def __init__(self, root, transform=None):
        self.root = root
        self.transform = transform
        self.image_paths = [
            os.path.join(root, fname)
            for fname in sorted(os.listdir(root))
            if fname.lower().endswith(self.IMG_EXTS)
        ]
        if len(self.image_paths) == 0:
            raise FileNotFoundError(
                f"No image files found in flat folder: {root}. "
                f"Expected one of extensions: {self.IMG_EXTS}"
            )

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        with Image.open(img_path) as img:
            img = img.convert("RGB")
        if self.transform is not None:
            img = self.transform(img)
        return img, 0


class CelebASampler:
    def __init__(self, batch_size, normalize=True, train=True, shuffle=True,
                 data_root="../data", image_size=64):
        self.batch_size = batch_size
        tfms = [
            transforms.CenterCrop(140),
            transforms.Resize(image_size),
        ]
        if train:
            tfms.append(transforms.RandomHorizontalFlip())
        tfms.append(transforms.ToTensor())
        if normalize:
            tfms.append(transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)))

        # Accept either a direct CelebA path or a generic data root.
        # Prefer explicit CelebA folders first; a generic data root may exist
        # (e.g. with MNIST/CIFAR metadata) but contain no image files for ImageFolder.
        candidate_roots = [
            os.path.join(data_root, "celeba", "img_align_celeba"),
            os.path.join(data_root, "celeba", "aligned_celeba"),
            os.path.join(data_root, "img_align_celeba"),
            os.path.join(data_root, "aligned_celeba"),
            data_root,
        ]
        image_root = next((p for p in candidate_roots if os.path.isdir(p)), None)
        if image_root is None:
            raise FileNotFoundError(
                "Could not locate CelebA images. Expected one of: "
                f"{candidate_roots}"
            )

        transform = transforms.Compose(tfms)
        try:
            self.dataset = datasets.ImageFolder(root=image_root, transform=transform)
        except FileNotFoundError:
            # Common CelebA layout on HPC: images directly under img_align_celeba/.
            self.dataset = FlatImageDataset(root=image_root, transform=transform)
        self.loader = DataLoader(
            self.dataset, batch_size=batch_size, shuffle=shuffle,
            drop_last=True, num_workers=2, pin_memory=True)
        self.iterator = iter(self.loader)

    def sample(self):
        try:
            data, _ = next(self.iterator)
        except StopIteration:
            self.iterator = iter(self.loader)
            data, _ = next(self.iterator)
        return data


class CelebAHQSampler:
    """Sampler for CelebA-HQ: flat folder of high-resolution face images.

    Unlike the standard aligned CelebA, CelebA-HQ images are already
    cropped and centred at high resolution, so no CenterCrop is applied.
    Images are simply resized to ``image_size`` and normalised to [-1, 1].
    """

    def __init__(self, batch_size, normalize=True, train=True, shuffle=True,
                 data_root="../data", image_size=256):
        self.batch_size = batch_size
        tfms = [transforms.Resize(image_size)]
        if train:
            tfms.append(transforms.RandomHorizontalFlip())
        tfms.append(transforms.ToTensor())
        if normalize:
            tfms.append(transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)))

        # Accept either the direct image folder or a parent containing
        # a celeba_hq_256 / celeba-hq-256 subfolder.
        candidate_roots = [
            data_root,
            os.path.join(data_root, "celeba_hq_256"),
            os.path.join(data_root, "celeba-hq-256"),
            os.path.join(data_root, "celeba_hq"),
        ]
        image_root = next((p for p in candidate_roots if os.path.isdir(p)), None)
        if image_root is None:
            raise FileNotFoundError(
                "Could not locate CelebA-HQ images. Expected one of: "
                f"{candidate_roots}"
            )

        transform = transforms.Compose(tfms)
        try:
            self.dataset = datasets.ImageFolder(root=image_root, transform=transform)
        except FileNotFoundError:
            self.dataset = FlatImageDataset(root=image_root, transform=transform)
        self.loader = DataLoader(
            self.dataset, batch_size=batch_size, shuffle=shuffle,
            drop_last=True, num_workers=2, pin_memory=True)
        self.iterator = iter(self.loader)

    def sample(self):
        try:
            data, _ = next(self.iterator)
        except StopIteration:
            self.iterator = iter(self.loader)
            data, _ = next(self.iterator)
        return data


# ============================================================
# Noise samplers
# ============================================================

class GaussianNoiseSampler:
    def __init__(self, dim):
        self.dim = dim

    def sample(self):
        return torch.randn(self.dim)


class UnifNoiseSampler:
    def __init__(self, dim, min_val=-1, max_val=1):
        self.dim = dim
        self.min_val = min_val
        self.max_val = max_val

    def sample(self):
        return self.min_val + (self.max_val - self.min_val) * torch.rand(self.dim)


# ============================================================
# Factory functions
# ============================================================

def get_size_dataset(args):
    """Return (C, H, W) for image datasets or (2,) for 2D datasets."""
    name = args.dataset.lower()
    if name in ("cifar", "cifar10"):
        return (3, 32, 32)
    elif name in ("celeba", "celeba_hq"):
        return (3, args.image_size, args.image_size)
    elif name == "mnist":
        return (1, 32, 32)
    elif name == "mnist28":
        return (1, 28, 28)
    elif is_2d_dataset(name):
        return (2,)
    raise ValueError(f"Unknown dataset: {args.dataset}")


def get_size_entire_training_dataset(args):
    """Return the total number of training samples (infinite for 2D samplers)."""
    name = args.dataset.lower()
    if name in ("cifar", "cifar10", "mnist", "mnist28"):
        return 60_000
    elif name == "celeba":
        return 202_599
    elif name == "celeba_hq":
        return 30_000
    elif is_2d_dataset(name):
        return float("inf")
    raise ValueError(f"Unknown dataset: {args.dataset}")


def get_sampler(args, test=False):
    """Return a data sampler for the given dataset."""
    name = args.dataset.lower()

    # 2D toy datasets
    if name == "circles":
        return Circles(args.batch_size)
    elif name == "three_rings":
        return ThreeRings(args.batch_size)
    elif name == "eight_gaussians":
        return EightGaussians(args.batch_size)
    elif name == "moons":
        return Moons(args.batch_size)

    # Image datasets
    normalize = not args.images_in_0_1
    data_root = getattr(args, "data_root", "./data")
    if name == "cifar10":
        return CIFAR10Sampler(
            args.batch_size, normalize=normalize,
            train=not test, shuffle=not test,
            data_root=os.path.join(data_root, "cifar10"))
    elif name == "celeba":
        return CelebASampler(
            args.batch_size, normalize=normalize,
            train=not test, shuffle=not test,
            data_root=data_root, image_size=args.image_size)
    elif name == "celeba_hq":
        return CelebAHQSampler(
            args.batch_size, normalize=normalize,
            train=not test, shuffle=not test,
            data_root=data_root, image_size=args.image_size)
    elif name == "mnist":
        return MNISTSampler(
            args.batch_size, normalize=normalize, resize_32=True,
            train=not test, shuffle=not test,
            data_root=data_root)
    elif name == "mnist28":
        return MNISTSampler(
            args.batch_size, normalize=normalize, resize_32=False,
            train=not test, shuffle=not test,
            data_root=data_root)
    raise ValueError(f"Unknown dataset: {args.dataset}")


def get_noise(args):
    """Return a noise sampler matching the generator's expected input.

    For ``unet``, ``ncsnpp``, ``ncsnpp_embed``: returns image-shaped noise
    (the latter two also require a separate ``nz``-dim vector; see train.py).
    For ``mlp_2d``: returns a flat (batch_size, nz) vector (nz=2 typical for 2D).
    For ``resnet_MMDGAN``, ``otm``: returns a flat (batch_size, nz) vector.
    """
    NoiseCls = UnifNoiseSampler if args.type_noise == "unif" else GaussianNoiseSampler

    if args.model_name in ("unet", "ncsnpp", "ncsnpp_embed"):
        # These generators take image-shaped noise
        return NoiseCls((args.batch_size,) + get_size_dataset(args))
    else:
        # Standard generators (resnet_MMDGAN, otm, mlp_2d) take a flat noise vector
        return NoiseCls((args.batch_size, args.nz))


def get_noise_dataset(args):
    """Return a noise sampler matching the dataset shape (for JKO initialisation)."""
    shape = (args.batch_size,) + get_size_dataset(args)
    if not is_2d_dataset(args.dataset) and args.images_in_0_1:
        return UnifNoiseSampler(shape, min_val=0, max_val=1)
    return GaussianNoiseSampler(shape)
