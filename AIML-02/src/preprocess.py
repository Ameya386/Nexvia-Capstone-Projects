"""Shared preprocessing so training and serving use IDENTICAL steps."""
import numpy as np
import torch
from PIL import Image

MEAN = (0.485, 0.456, 0.406)   # ImageNet statistics (the backbone was pre-trained on ImageNet)
STD = (0.229, 0.224, 0.225)


def normalize_batch(x):
    """x: float tensor (B,1,H,W) in [0,1] -> (B,3,H,W) ImageNet-normalised."""
    x = x.repeat(1, 3, 1, 1)
    mean = x.new_tensor(MEAN).view(1, 3, 1, 1)
    std = x.new_tensor(STD).view(1, 3, 1, 1)
    return (x - mean) / std


def resize_gray(gray, size):
    """uint8 (H,W) -> uint8 (size,size)."""
    resample = getattr(Image, "Resampling", Image).BILINEAR
    return np.asarray(Image.fromarray(gray).resize((size, size), resample))
