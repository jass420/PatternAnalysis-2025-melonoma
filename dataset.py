"""
dataset.py - Data Loading and Preprocessing for ISIC 2020 Melanoma Dataset

Contains:
- load_dicom_rgb: Function to load and preprocess DICOM medical images
- SiamesePairs: Dataset for generating image pairs for Siamese network training
- ImageDataset: Dataset for single images (used during evaluation)
- Utility functions for creating balanced pairs
"""

import random
import numpy as np
import pandas as pd
import pydicom
import cv2

import torch
from torch.utils.data import Dataset

import albumentations as A
from albumentations.pytorch import ToTensorV2


def load_dicom_rgb(path: str) -> np.ndarray:
    """
    Load DICOM file and convert to RGB image with memory-efficient processing.

    Steps:
    1. Load DICOM pixel array
    2. Downsample large images (>1024x1024) to prevent memory errors
    3. Convert grayscale to RGB (stack 3 channels)
    4. Normalize to [0, 1] range
    5. Convert to uint8 [0, 255]

    Args:
        path (str): Path to DICOM file

    Returns:
        np.ndarray: RGB image (H, W, 3) in uint8 format
    """
    # Read DICOM file
    ds = pydicom.dcmread(path)
    arr = ds.pixel_array

    # Downsample very large images FIRST to save memory
    # Original ISIC images can be 4000x6000 (~275MB each)
    if arr.shape[0] > 1024 or arr.shape[1] > 1024:
        scale = min(1024 / arr.shape[0], 1024 / arr.shape[1])
        new_h = int(arr.shape[0] * scale)
        new_w = int(arr.shape[1] * scale)
        arr = cv2.resize(arr, (new_w, new_h), interpolation=cv2.INTER_AREA)

    # Convert to float32 for normalization
    arr = arr.astype(np.float32)

    # Convert grayscale (H, W) to RGB (H, W, 3)
    if arr.ndim == 2:
        arr = np.stack([arr, arr, arr], axis=-1)

    # Normalize to [0, 1]
    arr -= arr.min()
    if arr.max() > 0:
        arr /= arr.max()

    # Convert to uint8 [0, 255]
    img = (arr * 255).clip(0, 255).astype(np.uint8)

    return img


def make_fixed_pairs(df, n_pos_each=1000, n_neg=3000, seed=42):
    """
    Create a fixed set of image pairs for validation (deterministic).

    Args:
        df (pd.DataFrame): DataFrame with image metadata
        n_pos_each (int): Number of positive pairs per class
        n_neg (int): Number of negative pairs (cross-class)
        seed (int): Random seed for reproducibility

    Returns:
        list: List of tuples (idx1, idx2, label)
              label=1.0 for same class, 0.0 for different class
    """
    rng = random.Random(seed)

    # Separate indices by class
    idx0 = df.index[df.target == 0].tolist()  # Normal
    idx1 = df.index[df.target == 1].tolist()  # Melanoma

    pairs = []

    def sample_pos(pool, n):
        """Sample positive pairs from a single class."""
        out = []
        if len(pool) < 2:
            return out
        for _ in range(n):
            i1, i2 = rng.sample(pool, 2)
            out.append((i1, i2, 1.0))
        return out

    # Positive pairs (same class)
    pairs += sample_pos(idx0, n_pos_each)  # Normal-Normal
    pairs += sample_pos(idx1, n_pos_each)  # Melanoma-Melanoma

    # Negative pairs (different classes)
    for _ in range(n_neg):
        i1 = rng.choice(idx0)  # Normal
        i2 = rng.choice(idx1)  # Melanoma
        pairs.append((i1, i2, 0.0))

    # Shuffle pairs
    rng.shuffle(pairs)

    return pairs


def make_random_pairs(df, n_pairs, seed=None):
    """
    Create random image pairs with balanced sampling for training.

    Balancing strategy:
    - 50% positive pairs (same class)
    - 50% negative pairs (different class)
    - For positive pairs, randomly select class to handle imbalance
      (ISIC 2020: 98.2% normal, 1.8% melanoma)

    Args:
        df (pd.DataFrame): DataFrame with image metadata
        n_pairs (int): Total number of pairs to generate
        seed (int, optional): Random seed

    Returns:
        list: List of tuples (idx1, idx2, label)
    """
    rng = random.Random(seed)

    # Separate indices by class
    idx0 = df.index[df.target == 0].tolist()  # Normal
    idx1 = df.index[df.target == 1].tolist()  # Melanoma

    pairs = []

    for _ in range(n_pairs):
        if rng.random() < 0.5:
            # Positive pair (same class)
            # Randomly select class for balance
            cls = rng.choice([0, 1])
            pool = idx0 if cls == 0 else idx1

            if len(pool) >= 2:
                i1, i2 = rng.sample(pool, 2)
            else:
                # Edge case: not enough samples in class
                i1 = i2 = pool[0]

            pairs.append((i1, i2, 1.0))
        else:
            # Negative pair (different classes)
            i1 = rng.choice(idx0)
            i2 = rng.choice(idx1)
            pairs.append((i1, i2, 0.0))

    # Shuffle pairs
    rng.shuffle(pairs)

    return pairs


class SiamesePairs(Dataset):
    """
    Dataset for Siamese Network training using image pairs.

    Each sample is a pair of images with a label indicating whether
    they belong to the same class (1.0) or different classes (0.0).

    Args:
        df (pd.DataFrame): DataFrame with columns ['dcm_path', 'target']
        pairs (list): List of (idx1, idx2, label) tuples
        size (int): Target image size (will be resized to size x size)
        augment (bool): Whether to apply data augmentation (for training)
    """

    def __init__(self, df, pairs, size=224, augment=True):
        self.df = df.reset_index(drop=True)
        self.pairs = pairs

        if augment:
            # Training augmentation pipeline
            self.transform = A.Compose([
                A.Resize(height=size, width=size),
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.Rotate(limit=20, p=0.5),
                A.RandomBrightnessContrast(
                    brightness_limit=0.2,
                    contrast_limit=0.2,
                    p=0.6
                ),
                A.HueSaturationValue(
                    hue_shift_limit=10,
                    sat_shift_limit=20,
                    val_shift_limit=10,
                    p=0.5
                ),
                A.GaussNoise(p=0.3),
                A.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                ),
                ToTensorV2()
            ])
        else:
            # Validation/test: minimal preprocessing
            self.transform = A.Compose([
                A.Resize(height=size, width=size),
                A.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                ),
                ToTensorV2()
            ])

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, i):
        """
        Get a pair of images.

        Returns:
            tuple: (img1_tensor, img2_tensor, label_tensor)
        """
        idx1, idx2, label = self.pairs[i]

        # Load images
        path1 = self.df.dcm_path.iloc[idx1]
        path2 = self.df.dcm_path.iloc[idx2]

        img1 = load_dicom_rgb(path1)
        img2 = load_dicom_rgb(path2)

        # Apply transforms
        x1 = self.transform(image=img1)["image"]
        x2 = self.transform(image=img2)["image"]

        # Convert label to tensor
        y = torch.tensor(label, dtype=torch.float32)

        return x1, x2, y


class ImageDataset(Dataset):
    """
    Dataset for single images (used during evaluation/inference).

    Args:
        df (pd.DataFrame): DataFrame with columns ['dcm_path', 'target']
        size (int): Target image size
    """

    def __init__(self, df, size=224):
        self.df = df.reset_index(drop=True)

        # Minimal preprocessing for evaluation
        self.transform = A.Compose([
            A.Resize(height=size, width=size),
            A.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            ),
            ToTensorV2()
        ])

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i):
        """
        Get a single image.

        Returns:
            tuple: (image_tensor, label_tensor)
        """
        row = self.df.iloc[i]

        # Load image
        img = load_dicom_rgb(row.dcm_path)

        # Apply transform
        x = self.transform(image=img)["image"]

        # Get label
        y = torch.tensor(row.target, dtype=torch.long)

        return x, y
