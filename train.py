"""
train.py - Training, Validation, Testing and Model Saving

This script trains the Siamese network on ISIC 2020 melanoma dataset,
validates during training, tests on validation set, and saves the best model.

Plots training/validation losses and metrics throughout training.
"""

import os
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt

# Import from custom modules
from modules import SiameseNet, ContrastiveLoss, PrototypeClassifier
from dataset import SiamesePairs, ImageDataset, make_fixed_pairs, make_random_pairs

# ============== CONFIGURATION ==============
BASE = Path(r"C:\Users\mjas0\OneDrive\Desktop\courses\COMP3710\Alzheimer-s")
CATALOG_CSV = BASE / "train_mapping.csv"
OUT_DIR = BASE / "outputs"
OUT_DIR.mkdir(exist_ok=True)

# Hyperparameters
IMG_SIZE = 224
EPOCHS = 12
BATCH_SIZE = 64
PAIRS_TRAIN = 6000
PAIRS_VAL = 1200
LR = 5e-4
MARGIN = 2.0
SEED = 42
NUM_WORKERS = 2
PREFETCH_FACTOR = 2

# Device
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {DEVICE}\n")

# Set random seeds
torch.manual_seed(SEED)
np.random.seed(SEED)


def split_train_val(df, test_size=0.2, seed=SEED):
    """
    Split data into train/validation sets grouped by patient_id.

    This prevents data leakage by ensuring samples from the same patient
    don't appear in both train and validation sets.
    """
    assert 'patient_id' in df.columns, "DataFrame must have 'patient_id' column"

    gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
    train_idx, val_idx = next(gss.split(df, groups=df['patient_id']))

    train_df = df.iloc[train_idx].reset_index(drop=True)
    val_df = df.iloc[val_idx].reset_index(drop=True)

    return train_df, val_df


@torch.no_grad()
def compute_embeddings(model, dataloader, device):
    """Compute embeddings for all samples in a dataloader."""
    model.eval()
    all_embeds = []
    all_labels = []

    for x, y in dataloader:
        x = x.to(device)
        z = model(x).cpu().numpy()
        all_embeds.append(z)
        all_labels.append(y.numpy())

    return np.vstack(all_embeds), np.concatenate(all_labels)


def train_one_epoch(model, dataloader, criterion, optimizer, scaler, device, epoch):
    """Train for one epoch."""
    model.train()
    total_loss = 0.0

    for batch_idx, (x1, x2, y) in enumerate(dataloader, 1):
        x1, x2, y = x1.to(device), x2.to(device), y.to(device)

        optimizer.zero_grad()

        # Mixed precision training
        with torch.amp.autocast(device, enabled=(device == "cuda")):
            z1 = model(x1)
            z2 = model(x2)
            loss = criterion(z1, z2, y)

        # Backward pass
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item()

        # Print progress every 10 batches
        if batch_idx % 10 == 0:
            print(f"  Batch {batch_idx}/{len(dataloader)} | Loss: {loss.item():.4f}")

    return total_loss / len(dataloader)


@torch.no_grad()
def validate(model, dataloader, criterion, device):
    """Validate the model."""
    model.eval()
    total_loss = 0.0

    for x1, x2, y in dataloader:
        x1, x2, y = x1.to(device), x2.to(device), y.to(device)

        with torch.amp.autocast(device, enabled=(device == "cuda")):
            z1 = model(x1)
            z2 = model(x2)
            loss = criterion(z1, z2, y)

        total_loss += loss.item()

    return total_loss / len(dataloader)


def plot_training_history(train_losses, val_losses, save_path):
    """Plot training and validation losses."""
    plt.figure(figsize=(10, 6))
    epochs_range = range(1, len(train_losses) + 1)

    plt.plot(epochs_range, train_losses, 'b-o', label='Training Loss', linewidth=2)
    plt.plot(epochs_range, val_losses, 'r-o', label='Validation Loss', linewidth=2)

    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Contrastive Loss', fontsize=12)
    plt.title('Training and Validation Losses', fontsize=14, fontweight='bold')
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Loss plot saved to: {save_path}")
    plt.close()


def plot_confusion_matrix(cm, save_path):
    """Plot confusion matrix."""
    fig, ax = plt.subplots(figsize=(8, 6))

    im = ax.imshow(cm, cmap='Blues')
    ax.figure.colorbar(im, ax=ax)

    # Labels
    classes = ['Normal', 'Melanoma']
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(classes)
    ax.set_yticklabels(classes)

    # Annotate cells
    for i in range(2):
        for j in range(2):
            text = ax.text(j, i, cm[i, j],
                          ha="center", va="center",
                          color="white" if cm[i, j] > cm.max() / 2 else "black",
                          fontsize=20, fontweight='bold')

    ax.set_xlabel('Predicted', fontsize=12)
    ax.set_ylabel('True', fontsize=12)
    ax.set_title('Confusion Matrix', fontsize=14, fontweight='bold')

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Confusion matrix plot saved to: {save_path}")
    plt.close()


def main():
    """Main training pipeline."""
    print("="*70)
    print("SIAMESE NETWORK TRAINING - ISIC 2020 MELANOMA CLASSIFICATION")
    print("="*70)
    print()

    # Load data
    print("Loading dataset...")
    df = pd.read_csv(CATALOG_CSV)
    train_df, val_df = split_train_val(df)

    print(f"Train samples: {len(train_df)}")
    print(f"Validation samples: {len(val_df)}")
    print()

    # Create validation pairs (fixed for consistent evaluation)
    val_pairs = make_fixed_pairs(val_df, n_pos_each=300, n_neg=600, seed=SEED)
    val_dataset = SiamesePairs(val_df, val_pairs, size=IMG_SIZE, augment=False)
    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        prefetch_factor=PREFETCH_FACTOR if NUM_WORKERS > 0 else None,
        persistent_workers=True if NUM_WORKERS > 0 else False
    )

    # Initialize model
    print("Initializing model...")
    model = SiameseNet(embed_dim=512).to(DEVICE)
    criterion = ContrastiveLoss(margin=MARGIN)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-6)
    scaler = torch.amp.GradScaler(DEVICE, enabled=(DEVICE == "cuda"))

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {total_params / 1e6:.2f}M")
    print()

    # Training loop
    print("="*70)
    print("TRAINING")
    print("="*70)
    print()

    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    best_model_path = OUT_DIR / "best_siamese.pth"

    for epoch in range(1, EPOCHS + 1):
        print(f"Epoch {epoch}/{EPOCHS}")
        print("-" * 70)

        # Create random training pairs for this epoch
        train_pairs = make_random_pairs(train_df, PAIRS_TRAIN, seed=SEED + epoch)
        train_dataset = SiamesePairs(train_df, train_pairs, size=IMG_SIZE, augment=True)
        train_loader = DataLoader(
            train_dataset,
            batch_size=BATCH_SIZE,
            shuffle=True,
            num_workers=NUM_WORKERS,
            pin_memory=True,
            drop_last=True,
            prefetch_factor=PREFETCH_FACTOR if NUM_WORKERS > 0 else None,
            persistent_workers=True if NUM_WORKERS > 0 else False
        )

        # Train
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, scaler, DEVICE, epoch)
        train_losses.append(train_loss)

        # Validate
        val_loss = validate(model, val_loader, criterion, DEVICE)
        val_losses.append(val_loss)

        # Learning rate step
        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]

        # Print epoch summary
        print(f"\nEpoch {epoch} Summary:")
        print(f"  Train Loss: {train_loss:.4f}")
        print(f"  Val Loss:   {val_loss:.4f}")
        print(f"  Learning Rate: {current_lr:.2e}")

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), best_model_path)
            print(f"  *** NEW BEST MODEL *** (val_loss={best_val_loss:.4f})")

        print("=" * 70)
        print()

    # Plot training history
    print("Plotting training history...")
    plot_training_history(train_losses, val_losses, OUT_DIR / "training_history.png")
    print()

    # ============== TESTING ==============
    print("="*70)
    print("TESTING ON VALIDATION SET")
    print("="*70)
    print()

    # Load best model
    model.load_state_dict(torch.load(best_model_path))
    model.eval()

    # Compute embeddings
    print("Computing embeddings...")
    train_subset = train_df.sample(n=min(3000, len(train_df)), random_state=SEED)
    test_subset = val_df.sample(n=min(2000, len(val_df)), random_state=SEED)

    train_dataset = ImageDataset(train_subset, size=IMG_SIZE)
    test_dataset = ImageDataset(test_subset, size=IMG_SIZE)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=False,
                             num_workers=NUM_WORKERS, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False,
                            num_workers=NUM_WORKERS, pin_memory=True)

    train_embeds, train_labels = compute_embeddings(model, train_loader, DEVICE)
    test_embeds, test_labels = compute_embeddings(model, test_loader, DEVICE)

    # Classify using prototypes
    print("Classifying test samples...")
    classifier = PrototypeClassifier()
    classifier.fit(train_embeds, train_labels)
    predictions = classifier.predict(test_embeds)

    # Compute metrics
    accuracy = accuracy_score(test_labels, predictions)
    cm = confusion_matrix(test_labels, predictions)

    print()
    print("="*70)
    print("TEST RESULTS")
    print("="*70)
    print(f"\nTest Accuracy: {accuracy:.4f} ({accuracy * 100:.2f}%)")
    print()
    print("Confusion Matrix:")
    print(cm)
    print("  [Normal→Normal, Normal→Melanoma]")
    print("  [Melanoma→Normal, Melanoma→Melanoma]")
    print()
    print("Classification Report:")
    print(classification_report(test_labels, predictions, target_names=['Normal', 'Melanoma']))
    print()

    # Plot confusion matrix
    plot_confusion_matrix(cm, OUT_DIR / "confusion_matrix.png")

    # Save results
    results = {
        'accuracy': float(accuracy),
        'confusion_matrix': cm.tolist(),
        'train_losses': train_losses,
        'val_losses': val_losses
    }

    import json
    with open(OUT_DIR / "training_results.json", 'w') as f:
        json.dump(results, f, indent=2)

    print("="*70)
    print(f"Training complete! Best model saved to: {best_model_path}")
    print("="*70)


if __name__ == "__main__":
    main()
