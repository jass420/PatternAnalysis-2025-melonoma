"""
Complete Siamese Network Training and Evaluation in ONE file.
Run this file and it will train the model and immediately show you the test accuracy.
"""
import os, random
from pathlib import Path
import numpy as np
import pandas as pd
import pydicom, cv2

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import models

import albumentations as A
from albumentations.pytorch import ToTensorV2
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# ============== CONFIG ==============
BASE        = Path(r"C:\Users\mjas0\OneDrive\Desktop\courses\COMP3710\Alzheimer-s")
CATALOG_CSV = BASE / "train_mapping.csv"
OUT_DIR     = BASE / "outputs"; OUT_DIR.mkdir(exist_ok=True)

IMG_SIZE        = 224       # Good balance of quality and speed
EPOCHS          = 15        # Good balance for convergence
BATCH_SIZE      = 64        # REDUCED for faster training (still good GPU usage)
PAIRS_TRAIN     = 6000      # REDUCED for faster epochs but still enough data
PAIRS_VAL       = 1200      # REDUCED proportionally
LR              = 5e-4
MARGIN          = 2.0
SEED            = 42
NUM_WORKERS     = 3         # REDUCED to avoid memory errors
DEVICE          = "cuda" if torch.cuda.is_available() else "cpu"
PREFETCH_FACTOR = 4         # REDUCED (less memory overhead)

# For evaluation - use MORE samples for better accuracy
EVAL_TRAIN_SUBSET = 3000    # INCREASED from 1000
EVAL_TEST_SUBSET = 2000     # INCREASED from 1000
# ====================================

random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)

# ---------- DICOM loader ----------
def load_dicom_rgb(path: str) -> np.ndarray:
    """Load DICOM and convert to RGB with memory-efficient processing."""
    ds = pydicom.dcmread(path)
    arr = ds.pixel_array

    # Resize FIRST to save memory (before float conversion)
    if arr.shape[0] > 1024 or arr.shape[1] > 1024:
        # Downsample very large images first
        scale = min(1024 / arr.shape[0], 1024 / arr.shape[1])
        new_h = int(arr.shape[0] * scale)
        new_w = int(arr.shape[1] * scale)
        arr = cv2.resize(arr, (new_w, new_h), interpolation=cv2.INTER_AREA)

    arr = arr.astype(np.float32)
    if arr.ndim == 2:
        arr = np.stack([arr, arr, arr], axis=-1)
    arr -= arr.min()
    if arr.max() > 0: arr /= arr.max()
    img = (arr * 255).clip(0, 255).astype(np.uint8)
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# ---------- Hard Negative Mining (COMMENTED OUT - decreased accuracy) ----------
# @torch.no_grad()
# def mine_hard_pairs(model, df, device, top_k=2000):
#     """
#     Mine hard pairs by finding:
#     - Hard negatives: Different classes but close in embedding space
#     - Hard positives: Same class but far in embedding space
#     """
#     print("  [Mining hard pairs...]")
#
#     # Prepare dataset and loader
#     dataset = ImageDataset(df, size=IMG_SIZE)
#     loader = DataLoader(dataset, batch_size=128, shuffle=False, num_workers=2, pin_memory=True)
#
#     # Compute all embeddings
#     model.eval()
#     all_embeds = []
#     all_labels = []
#     for x, y in loader:
#         x = x.to(device)
#         z = model(x).cpu().numpy()
#         all_embeds.append(z)
#         all_labels.append(y.numpy())
#
#     embeds = np.vstack(all_embeds)
#     labels = np.concatenate(all_labels)
#
#     # Find hard pairs
#     hard_pairs = []
#
#     # Sample a subset for efficiency
#     n_samples = min(5000, len(df))
#     sample_indices = np.random.choice(len(df), n_samples, replace=False)
#
#     for i in sample_indices[:1000]:
#         dists = np.linalg.norm(embeds - embeds[i], axis=1)
#
#         # Hard negatives: different class but close (distance < 1.5)
#         diff_class_mask = labels != labels[i]
#         diff_class_indices = np.where(diff_class_mask)[0]
#         if len(diff_class_indices) > 0:
#             diff_class_dists = dists[diff_class_indices]
#             hard_neg_idx = diff_class_indices[diff_class_dists < 1.5]
#             for j in hard_neg_idx[:3]:
#                 hard_pairs.append((i, j, 0.0))
#
#         # Hard positives: same class but far (distance > 0.8)
#         same_class_mask = (labels == labels[i]) & (np.arange(len(labels)) != i)
#         same_class_indices = np.where(same_class_mask)[0]
#         if len(same_class_indices) > 0:
#             same_class_dists = dists[same_class_indices]
#             hard_pos_idx = same_class_indices[same_class_dists > 0.8]
#             for j in hard_pos_idx[:2]:
#                 hard_pairs.append((i, j, 1.0))
#
#     print(f"  [Found {len(hard_pairs)} hard pairs]")
#     return hard_pairs
#
# def make_hard_augmented_pairs(df, n_pairs, hard_pairs, hard_ratio=0.5, seed=None):
#     """
#     Create pairs with mix of hard mined pairs and random pairs.
#     hard_ratio: Fraction of pairs that should be hard (0.5 = 50% hard, 50% random)
#     """
#     n_hard = int(n_pairs * hard_ratio)
#     n_random = n_pairs - n_hard
#
#     # Sample hard pairs
#     if len(hard_pairs) > 0:
#         hard_sample_indices = np.random.choice(len(hard_pairs), min(n_hard, len(hard_pairs)), replace=False)
#         sampled_hard = [hard_pairs[i] for i in hard_sample_indices]
#         while len(sampled_hard) < n_hard:
#             sampled_hard.append(random.choice(hard_pairs))
#     else:
#         sampled_hard = []
#         n_random = n_pairs
#
#     # Create random pairs
#     random_pairs = make_random_pairs(df, n_random, seed=seed)
#
#     # Combine and shuffle
#     all_pairs = sampled_hard + random_pairs
#     random.shuffle(all_pairs)
#     return all_pairs

# ---------- Pair-making helpers ----------
def make_fixed_pairs(df, n_pos_each=300, n_neg=300, seed=SEED):
    rng = random.Random(seed)
    idx0 = df.index[df.target == 0].tolist()
    idx1 = df.index[df.target == 1].tolist()
    pairs = []

    def sample_pos(pool, n):
        out = []
        if len(pool) < 2: return out
        for _ in range(n):
            i1, i2 = rng.sample(pool, 2)
            out.append((i1, i2, 1.0))
        return out

    pairs += sample_pos(idx0, n_pos_each)
    pairs += sample_pos(idx1, n_pos_each)
    for _ in range(n_neg):
        i1 = rng.choice(idx0); i2 = rng.choice(idx1)
        pairs.append((i1, i2, 0.0))
    rng.shuffle(pairs)
    return pairs

def make_random_pairs(df, n_pairs, seed=None):
    rng = random.Random(seed)
    idx0 = df.index[df.target == 0].tolist()
    idx1 = df.index[df.target == 1].tolist()
    pairs = []
    for _ in range(n_pairs):
        if rng.random() < 0.5:
            cls = rng.choice([0, 1])
            pool = idx0 if cls == 0 else idx1
            if len(pool) >= 2:
                i1, i2 = rng.sample(pool, 2)
            else:
                i1 = i2 = pool[0]
            pairs.append((i1, i2, 1.0))
        else:
            i1 = rng.choice(idx0); i2 = rng.choice(idx1)
            pairs.append((i1, i2, 0.0))
    rng.shuffle(pairs)
    return pairs

# ---------- Datasets ----------
class SiamesePairs(Dataset):
    def __init__(self, df, pairs, size=224, augment=True):
        self.df = df.reset_index(drop=True)
        self.pairs = pairs
        if augment:
            tf = [
                A.Resize(height=size, width=size),
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.Rotate(limit=20, p=0.5),
                A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.6),
                A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=20, val_shift_limit=10, p=0.5),
                A.GaussNoise(p=0.3),
                A.Normalize(), ToTensorV2()
            ]
        else:
            tf = [A.Resize(height=size, width=size), A.Normalize(), ToTensorV2()]
        self.tf = A.Compose(tf)

    def __len__(self): return len(self.pairs)

    def __getitem__(self, i):
        i1, i2, y = self.pairs[i]
        p1 = self.df.dcm_path.iloc[i1]
        p2 = self.df.dcm_path.iloc[i2]
        img1 = load_dicom_rgb(p1); img2 = load_dicom_rgb(p2)
        x1 = self.tf(image=img1)["image"]
        x2 = self.tf(image=img2)["image"]
        return x1, x2, torch.tensor(y, dtype=torch.float32)

class ImageDataset(Dataset):
    """For evaluation - single images"""
    def __init__(self, df, size=224):
        self.df = df.reset_index(drop=True)
        self.tf = A.Compose([
            A.Resize(height=size, width=size),
            A.Normalize(),
            ToTensorV2()
        ])

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i):
        row = self.df.iloc[i]
        img = load_dicom_rgb(row.dcm_path)
        x = self.tf(image=img)["image"]
        y = torch.tensor(row.target, dtype=torch.long)
        return x, y

# ---------- Model ----------
class SiameseNet(nn.Module):
    def __init__(self, embed_dim=512):  # LARGER embeddings
        super().__init__()
        try:
            # USE ResNet50 - MUCH DEEPER for more GPU work!
            backbone = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
        except Exception:
            backbone = models.resnet50(pretrained=True)
        backbone.fc = nn.Identity()
        self.backbone = backbone

        # MUCH DEEPER projection head with MORE layers = MORE GPU computation!
        self.head = nn.Sequential(
            nn.Linear(2048, 2048),  # ResNet50 outputs 2048 features
            nn.BatchNorm1d(2048),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),

            nn.Linear(2048, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),

            nn.Linear(1024, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),

            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),

            nn.Linear(512, embed_dim)
        )

    def forward(self, x):
        f = self.backbone(x)
        z = self.head(f)
        return F.normalize(z, dim=1)

def contrastive_loss(z1, z2, y, margin=MARGIN):
    d = F.pairwise_distance(z1, z2)
    pos = y * (d ** 2)
    neg = (1 - y) * (F.relu(margin - d) ** 2)
    return (pos + neg).mean()

# ---------- Split ----------
def split_train_val(df, seed=SEED):
    assert 'patient_id' in df.columns
    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=seed)
    tr_idx, va_idx = next(gss.split(df, groups=df['patient_id']))
    return df.iloc[tr_idx].reset_index(drop=True), df.iloc[va_idx].reset_index(drop=True)

# ---------- Training ----------
def train_model(train_df, val_df):
    print("="*70)
    print("STEP 1: TRAINING SIAMESE NETWORK")
    print("="*70)
    print()

    val_pairs = make_fixed_pairs(val_df, n_pos_each=300, n_neg=300, seed=SEED)
    val_ds = SiamesePairs(val_df, val_pairs, size=IMG_SIZE, augment=False)
    val_dl = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False,
                        num_workers=NUM_WORKERS, pin_memory=True,
                        prefetch_factor=PREFETCH_FACTOR if NUM_WORKERS > 0 else None,
                        persistent_workers=True if NUM_WORKERS > 0 else False)

    net = SiameseNet(embed_dim=512).to(DEVICE)  # Match new embedding size
    opt = torch.optim.AdamW(net.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS, eta_min=1e-6)
    scaler = torch.amp.GradScaler(DEVICE, enabled=(DEVICE == "cuda"))

    print(f"\nModel Parameters: {sum(p.numel() for p in net.parameters()) / 1e6:.2f}M")
    print(f"GPU will process {BATCH_SIZE} image pairs simultaneously\n")

    best_val = float("inf")
    best_path = OUT_DIR / "best.pt"

    print(f"Training on {DEVICE}")
    print(f"Epochs: {EPOCHS} | Pairs per epoch: {PAIRS_TRAIN}\n")

    for epoch in range(1, EPOCHS+1):
        print(f"\n{'='*70}")
        print(f"EPOCH {epoch}/{EPOCHS}")
        print(f"{'='*70}")

        train_pairs = make_random_pairs(train_df, PAIRS_TRAIN, seed=SEED + epoch)
        train_ds = SiamesePairs(train_df, train_pairs, size=IMG_SIZE, augment=True)
        train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=NUM_WORKERS, pin_memory=True, drop_last=True,
                              prefetch_factor=PREFETCH_FACTOR if NUM_WORKERS > 0 else None,
                              persistent_workers=True if NUM_WORKERS > 0 else False)

        # Train
        print(f"[Training] Processing {len(train_dl)} batches...")
        net.train()
        tr_loss = 0.0
        for batch_idx, (x1, x2, y) in enumerate(train_dl, 1):
            x1, x2, y = x1.to(DEVICE), x2.to(DEVICE), y.to(DEVICE)
            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast(DEVICE, enabled=(DEVICE=="cuda")):
                z1, z2 = net(x1), net(x2)
                loss = contrastive_loss(z1, z2, y)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            tr_loss += loss.item()

            # Show progress every 10 batches
            if batch_idx % 10 == 0 or batch_idx == 1:
                print(f"  Batch {batch_idx}/{len(train_dl)} | Loss: {loss.item():.4f}")

        tr_loss /= len(train_dl)

        # Validate
        print(f"[Validation] Processing {len(val_dl)} batches...")
        net.eval()
        va_loss = 0.0
        with torch.no_grad():
            for x1, x2, y in val_dl:
                x1, x2, y = x1.to(DEVICE), x2.to(DEVICE), y.to(DEVICE)
                z1, z2 = net(x1), net(x2)
                va_loss += contrastive_loss(z1, z2, y).item()
        va_loss /= len(val_dl)

        print()
        print(f"EPOCH {epoch}/{EPOCHS} SUMMARY:")
        print(f"  Train Loss: {tr_loss:.4f}")
        print(f"  Val Loss:   {va_loss:.4f}")

        torch.save(net.state_dict(), OUT_DIR / "last.pt")
        if va_loss < best_val:
            best_val = va_loss
            torch.save(net.state_dict(), best_path)
            print(f"  *** NEW BEST MODEL *** (val_loss={best_val:.4f})")
        else:
            print(f"  (Best so far: {best_val:.4f})")

        # Step learning rate scheduler
        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]
        print(f"  Learning rate: {current_lr:.2e}")
        print(f"{'='*70}")

    print("Training complete!\n")
    return net

# ---------- Evaluation ----------
@torch.no_grad()
def compute_embeddings(model, dataloader, device, desc="Computing"):
    model.eval()
    all_embeds = []
    all_labels = []
    print(f"{desc}... ({len(dataloader)} batches)", flush=True)
    for batch_idx, (x, y) in enumerate(dataloader, 1):
        x = x.to(device)
        z = model(x).cpu().numpy()
        all_embeds.append(z)
        all_labels.append(y.numpy())
        if batch_idx % 5 == 0:
            print(f"  Batch {batch_idx}/{len(dataloader)}", flush=True)
    return np.vstack(all_embeds), np.concatenate(all_labels)

def classify_by_prototypes(train_embeds, train_labels, test_embeds):
    """
    Classification: Find which class prototype is closest to each test sample.
    """
    unique_classes = np.unique(train_labels)
    prototypes = {}
    for c in unique_classes:
        prototypes[c] = train_embeds[train_labels == c].mean(axis=0)

    preds = []
    for emb in test_embeds:
        dists = {c: np.linalg.norm(emb - proto) for c, proto in prototypes.items()}
        pred_class = min(dists, key=dists.get)
        preds.append(pred_class)
    return np.array(preds)

def evaluate_model(net, train_df, val_df):
    print("="*70)
    print("STEP 2: EVALUATING ON TEST SET")
    print("="*70)
    print()

    # Use subsets for fast evaluation
    train_subset = train_df.sample(n=min(EVAL_TRAIN_SUBSET, len(train_df)), random_state=SEED)
    test_subset = val_df.sample(n=min(EVAL_TEST_SUBSET, len(val_df)), random_state=SEED)

    print(f"Using {len(train_subset)} train samples for prototypes")
    print(f"Using {len(test_subset)} test samples for evaluation\n")

    train_ds = ImageDataset(train_subset, size=IMG_SIZE)
    test_ds = ImageDataset(test_subset, size=IMG_SIZE)

    train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=False,
                          num_workers=NUM_WORKERS, pin_memory=True)
    test_dl = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False,
                         num_workers=NUM_WORKERS, pin_memory=True)

    train_embeds, train_labels = compute_embeddings(net, train_dl, DEVICE, "Computing train embeddings")
    test_embeds, test_labels = compute_embeddings(net, test_dl, DEVICE, "Computing test embeddings")

    print("\nClassifying test samples...")
    preds = classify_by_prototypes(train_embeds, train_labels, test_embeds)
    print("Classification complete!")

    acc = accuracy_score(test_labels, preds)

    print()
    print("="*70)
    print(f"TEST ACCURACY: {acc:.4f} ({acc*100:.2f}%)")
    print("="*70)
    print()

    print("Classification Report:")
    print(classification_report(test_labels, preds, target_names=['Normal', 'Melanoma'], digits=4))

    print("\nConfusion Matrix:")
    cm = confusion_matrix(test_labels, preds)
    print(cm)
    print("  [Normal→Normal, Normal→Melanoma]")
    print("  [Melanoma→Normal, Melanoma→Melanoma]")
    print()

# ---------- Main ----------
def main():
    import time
    start_time = time.time()

    print("="*70)
    print("SIAMESE NETWORK: TRAINING + EVALUATION")
    print("="*70)
    print()

    df = pd.read_csv(CATALOG_CSV)
    train_df, val_df = split_train_val(df)

    print(f"Dataset: {len(train_df)} train, {len(val_df)} test")
    print()

    # Step 1: Train
    train_start = time.time()
    net = train_model(train_df, val_df)
    train_time = time.time() - train_start

    # Step 2: Evaluate
    eval_start = time.time()
    evaluate_model(net, train_df, val_df)
    eval_time = time.time() - eval_start

    total_time = time.time() - start_time

    print()
    print("="*70)
    print("ALL DONE!")
    print("="*70)
    print()
    print(f"Training time:   {train_time//60:.0f}m {train_time%60:.0f}s")
    print(f"Evaluation time: {eval_time//60:.0f}m {eval_time%60:.0f}s")
    print(f"Total time:      {total_time//60:.0f}m {total_time%60:.0f}s")
    print("="*70)

if __name__ == "__main__":
    main()
