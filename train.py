# train_only.py
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

# ============== CONFIG ==============
BASE        = Path(r"C:\Users\mjas0\OneDrive\Desktop\courses\COMP3710\Alzheimer-s")
CATALOG_CSV = BASE / "train_mapping.csv"  # built earlier
OUT_DIR     = BASE / "outputs"; OUT_DIR.mkdir(exist_ok=True)

IMG_SIZE        = 256
EPOCHS          = 10
BATCH_SIZE      = 32
PAIRS_TRAIN     = 25000     # pairs per epoch (train)
PAIRS_VAL       = 5000      # fixed val pairs
LR              = 1e-3
MARGIN          = 1.0
SEED            = 42
NUM_WORKERS     = 0         # Windows-safe; raise if stable
DEVICE          = "cuda" if torch.cuda.is_available() else "cpu"
# ====================================

random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)

# ---------- DICOM loader ----------
def load_dicom_rgb(path: str) -> np.ndarray:
    ds = pydicom.dcmread(path)
    arr = ds.pixel_array.astype(np.float32)
    if arr.ndim == 2:
        arr = np.stack([arr, arr, arr], axis=-1)
    arr -= arr.min()
    if arr.max() > 0: arr /= arr.max()
    img = (arr * 255).clip(0, 255).astype(np.uint8)
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # ensure RGB

# ---------- Pair-making helpers ----------
def make_fixed_pairs(df, n_pos_each=1000, n_neg=3000, seed=SEED):
    """Build a deterministic list of pairs for validation."""
    rng = random.Random(seed)
    idx0 = df.index[df.target == 0].tolist()
    idx1 = df.index[df.target == 1].tolist()
    pairs = []
    # positives: balanced across classes (as much as possible)
    def sample_pos(pool, n):
        out = []
        if len(pool) < 2:
            return out
        for _ in range(n):
            i1, i2 = rng.sample(pool, 2)
            out.append((i1, i2, 1.0))
        return out

    pairs += sample_pos(idx0, n_pos_each)
    pairs += sample_pos(idx1, n_pos_each)
    # negatives
    for _ in range(n_neg):
        i1 = rng.choice(idx0); i2 = rng.choice(idx1)
        pairs.append((i1, i2, 0.0))
    rng.shuffle(pairs)
    return pairs

def make_random_pairs(df, n_pairs, seed=None):
    """Fresh random pairs for training each epoch."""
    rng = random.Random(seed)
    idx0 = df.index[df.target == 0].tolist()
    idx1 = df.index[df.target == 1].tolist()
    pairs = []
    for _ in range(n_pairs):
        if rng.random() < 0.5:
            # positive: pick a class first to balance
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
    def __init__(self, df, pairs, size=256, augment=True):
        self.df = df.reset_index(drop=True)
        self.pairs = pairs
        if augment:
            tf = [
                A.RandomResizedCrop(size, size, scale=(0.85,1.0), ratio=(0.9,1.1)),
                A.HorizontalFlip(p=0.5), A.VerticalFlip(p=0.5),
                A.ShiftScaleRotate(0.05, 0.15, 25, p=0.7),
                A.RandomBrightnessContrast(p=0.5),
                A.CLAHE(p=0.2),
                A.Normalize(), ToTensorV2()
            ]
        else:
            tf = [A.Resize(size, size), A.Normalize(), ToTensorV2()]
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

# ---------- Model ----------
class SiameseNet(nn.Module):
    def __init__(self, embed_dim=128):
        super().__init__()
        try:
            backbone = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        except Exception:
            backbone = models.resnet18(pretrained=True)
        backbone.fc = nn.Identity()
        self.backbone = backbone
        self.head = nn.Sequential(
            nn.Linear(512, 512), nn.ReLU(inplace=True),
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
    assert 'patient_id' in df.columns, "train_mapping.csv needs 'patient_id'."
    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=seed)
    tr_idx, va_idx = next(gss.split(df, groups=df['patient_id']))
    return df.iloc[tr_idx].reset_index(drop=True), df.iloc[va_idx].reset_index(drop=True)

# ---------- Training ----------
def main():
    df = pd.read_csv(CATALOG_CSV)
    need = {'image_name','dcm_path','target','patient_id'}
    miss = need - set(df.columns)
    if miss:
        raise SystemExit(f"Missing columns in train_mapping.csv: {miss}")

    train_df, val_df = split_train_val(df)
    print(f"train: {len(train_df)}   val: {len(val_df)}")
    print("class balance (train):", dict(train_df['target'].value_counts()))

    # fixed val pairs (deterministic)
    val_pairs = make_fixed_pairs(val_df,
                                 n_pos_each=max(500, PAIRS_VAL//4),
                                 n_neg=max(1000, PAIRS_VAL//2),
                                 seed=SEED)
    val_ds = SiamesePairs(val_df, val_pairs, size=IMG_SIZE, augment=False)
    val_dl = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False,
                        num_workers=NUM_WORKERS, pin_memory=True)

    net = SiameseNet(embed_dim=128).to(DEVICE)
    opt = torch.optim.AdamW(net.parameters(), lr=LR)
    scaler = torch.cuda.amp.GradScaler(enabled=(DEVICE == "cuda"))

    best_val = float("inf"); best_path = OUT_DIR / "best.pt"

    for epoch in range(1, EPOCHS+1):
        # fresh random pairs each epoch for training
        train_pairs = make_random_pairs(train_df, PAIRS_TRAIN, seed=SEED + epoch)
        train_ds = SiamesePairs(train_df, train_pairs, size=IMG_SIZE, augment=True)
        train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=NUM_WORKERS, pin_memory=True, drop_last=True)

        # ---- train ----
        net.train()
        tr_loss = 0.0
        for x1, x2, y in train_dl:
            x1 = x1.to(DEVICE, non_blocking=True)
            x2 = x2.to(DEVICE, non_blocking=True)
            y  = y.to(DEVICE, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=(DEVICE=="cuda")):
                z1, z2 = net(x1), net(x2)
                loss = contrastive_loss(z1, z2, y)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            tr_loss += loss.item()
        tr_loss /= len(train_dl)

        # ---- validate (fixed pairs) ----
        net.eval()
        va_loss = 0.0
        with torch.no_grad():
            for x1, x2, y in val_dl:
                x1 = x1.to(DEVICE, non_blocking=True)
                x2 = x2.to(DEVICE, non_blocking=True)
                y  = y.to(DEVICE, non_blocking=True)
                z1, z2 = net(x1), net(x2)
                va_loss += contrastive_loss(z1, z2, y).item()
        va_loss /= len(val_dl)

        print(f"Epoch {epoch:02d}/{EPOCHS}  train_loss={tr_loss:.4f}  val_loss={va_loss:.4f}")

        # checkpointing
        torch.save(net.state_dict(), OUT_DIR / "last.pt")
        if va_loss < best_val:
            best_val = va_loss
            torch.save(net.state_dict(), best_path)
            print(f"  🔥 new best (val_loss={best_val:.4f}) → {best_path}")

    print(f"Done. Best val loss: {best_val:.4f}. Weights at {best_path} and {OUT_DIR/'last.pt'}.")

if __name__ == "__main__":
    main()
