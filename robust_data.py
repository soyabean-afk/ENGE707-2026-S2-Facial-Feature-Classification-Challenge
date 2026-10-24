"""Data pipeline for the round-2 ("robust") training run.

Kept in a .py file (not inside the notebook) so that DataLoader worker processes can import it on Windows:
worker processes cannot see classes defined inside a notebook, so with the notebook version images had to be
loaded one at a time (num_workers=0). Importing from this module allows several workers in parallel.
"""
import random

import numpy as np
import torch
from PIL import Image, ImageEnhance, ImageFilter
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms as T

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
MEAN_RGB = tuple(int(round(m * 255)) for m in IMAGENET_MEAN)   # padding colour (becomes ~0 after Normalize)


class ResizePad:
    """Resize so the longer side equals `size` (keeps aspect ratio), then pad to a size x size square."""

    def __init__(self, size, fill=MEAN_RGB):
        self.size, self.fill = size, fill

    def __call__(self, img):
        w, h = img.size
        scale = self.size / max(w, h)
        nw, nh = max(1, round(w * scale)), max(1, round(h * scale))
        img = img.resize((nw, nh), Image.BICUBIC)
        canvas = Image.new("RGB", (self.size, self.size), self.fill)
        canvas.paste(img, ((self.size - nw) // 2, (self.size - nh) // 2))
        return canvas

    def __repr__(self):
        return f"ResizePad(size={self.size})"


class RandomLowRes:
    """With probability p, shrink the image so its longer side is min_side..max_side pixels, then scale it back up.
    Simulates the tiny, pixelated crops found in the test set."""

    def __init__(self, p=0.25, min_side=10, max_side=40):
        self.p, self.min_side, self.max_side = p, min_side, max_side

    def __call__(self, img):
        if random.random() >= self.p:
            return img
        w, h = img.size
        target = random.randint(self.min_side, self.max_side)
        if max(w, h) <= target:
            return img
        s = target / max(w, h)
        small = img.resize((max(1, round(w * s)), max(1, round(h * s))),
                           random.choice([Image.NEAREST, Image.BILINEAR, Image.BICUBIC]))
        return small.resize((w, h), random.choice([Image.NEAREST, Image.BILINEAR]))

    def __repr__(self):
        return f"RandomLowRes(p={self.p}, side={self.min_side}-{self.max_side})"


class RandomDarken:
    """With probability p, multiply brightness by a factor in [low, high]. Simulates very dark test crops."""

    def __init__(self, p=0.15, low=0.15, high=0.5):
        self.p, self.low, self.high = p, low, high

    def __call__(self, img):
        if random.random() >= self.p:
            return img
        return ImageEnhance.Brightness(img).enhance(random.uniform(self.low, self.high))

    def __repr__(self):
        return f"RandomDarken(p={self.p}, factor={self.low}-{self.high})"


class RandomBlur:
    """With probability p, apply a Gaussian blur with a random radius (works on any image size)."""

    def __init__(self, p=0.15, max_radius=1.5):
        self.p, self.max_radius = p, max_radius

    def __call__(self, img):
        if random.random() >= self.p:
            return img
        return img.filter(ImageFilter.GaussianBlur(radius=random.uniform(0.3, self.max_radius)))

    def __repr__(self):
        return f"RandomBlur(p={self.p}, radius<={self.max_radius})"


class Degrade:
    """Fixed (non-random) degradation used for the robustness stress test on validation images."""

    def __init__(self, kind):
        assert kind in {"lowres", "dark", "blur"}
        self.kind = kind

    def __call__(self, img):
        w, h = img.size
        if self.kind == "lowres":                       # longer side 20 px, then back to the original size
            s = 20 / max(w, h)
            if s < 1:
                img = img.resize((max(1, round(w * s)), max(1, round(h * s))), Image.BILINEAR).resize((w, h), Image.BILINEAR)
            return img
        if self.kind == "dark":
            return ImageEnhance.Brightness(img).enhance(0.3)
        return img.filter(ImageFilter.GaussianBlur(radius=1.5))

    def __repr__(self):
        return f"Degrade({self.kind})"


def build_transforms(size, mode):
    """mode: "train" (robust augmentation), "eval" (validation/test), or "lowres"/"dark"/"blur" (stress test).
    Colour/blur/geometry augmentations run on the small original image (cheaper), before resizing to `size`."""
    norm = [T.ToTensor(), T.Normalize(IMAGENET_MEAN, IMAGENET_STD)]
    if mode == "eval":
        return T.Compose([ResizePad(size)] + norm)
    if mode in {"lowres", "dark", "blur"}:
        return T.Compose([Degrade(mode), ResizePad(size)] + norm)
    assert mode == "train"
    return T.Compose([
        RandomLowRes(p=0.25),
        RandomDarken(p=0.15),
        RandomBlur(p=0.15),
        T.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.03),
        T.RandomGrayscale(p=0.05),
        T.RandomHorizontalFlip(p=0.5),
        T.RandomAffine(degrees=15, translate=(0.08, 0.08), scale=(0.85, 1.15), fill=MEAN_RGB),
        ResizePad(size),
        *norm,
        T.RandomErasing(p=0.25, scale=(0.02, 0.15), value=0),   # hide a random patch (occlusion)
    ])


class FacialFeatureDataset(Dataset):
    """Returns (image_tensor, label) for labelled data, or (image_tensor, image_id) when labels is None."""

    def __init__(self, paths, ids, labels, transform, size):
        self.paths = [str(p) for p in paths]
        self.ids = [int(i) for i in ids]
        self.labels = None if labels is None else [int(l) for l in labels]
        self.transform = transform
        self.size = size

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        try:
            with Image.open(self.paths[i]) as im:
                img = im.convert("RGB")
        except Exception:
            img = Image.new("RGB", (self.size, self.size), MEAN_RGB)   # unreadable file: still gets a prediction
        x = self.transform(img)
        return x, (self.ids[i] if self.labels is None else self.labels[i])


def seed_worker(worker_id):
    s = torch.initial_seed() % 2**32
    np.random.seed(s)
    random.seed(s)


def make_loader(ds, shuffle, batch_size, num_workers, seed=42, pin_memory=False, persistent=False):
    # persistent=True keeps the worker processes alive between epochs (used for the training loader only, so
    # idle workers of the other loaders don't hold memory).
    g = torch.Generator()
    g.manual_seed(seed)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, drop_last=shuffle, num_workers=num_workers,
                      pin_memory=pin_memory, persistent_workers=persistent and num_workers > 0,
                      worker_init_fn=seed_worker, generator=g)