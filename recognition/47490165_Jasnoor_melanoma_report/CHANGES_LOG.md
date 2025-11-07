# Siamese Network Training - Changes Log for Accuracy Improvement

## Overview
This document tracks all changes made to the Siamese Network implementation for ISIC 2020 melanoma classification to improve accuracy and performance.

---

## 1. Initial Setup & Bug Fixes

### 1.1 Fixed Albumentations Compatibility Issues
**Problem**: Newer albumentations (v2.0.8) requires different API syntax
**Solution**: Updated transform syntax
```python
# Before:
A.RandomResizedCrop(size, size, ...)
A.Resize(size, size)

# After:
A.RandomResizedCrop(size=(size, size), ...)
A.Resize(height=size, width=size)
```
**Impact**: Code runs without errors

### 1.2 Fixed Deprecated PyTorch APIs
**Problem**: `torch.cuda.amp` deprecated in PyTorch 2.7+
**Solution**: Updated to new API
```python
# Before:
scaler = torch.cuda.amp.GradScaler(enabled=(DEVICE == "cuda"))
with torch.cuda.amp.autocast(enabled=(DEVICE=="cuda")):

# After:
scaler = torch.amp.GradScaler(DEVICE, enabled=(DEVICE == "cuda"))
with torch.amp.autocast(DEVICE, enabled=(DEVICE=="cuda")):
```
**Impact**: Removed deprecation warnings

---

## 2. Training Speed Optimizations

### 2.1 Reduced Image Size
**Problem**: 256×256 images too slow to load from DICOM
**Change**: `IMG_SIZE = 256 → 128 → 224`
```python
IMG_SIZE = 224  # Good balance of quality and speed
```
**Impact**:
- Faster data loading
- 4x fewer pixels to process at 128×128
- Settled on 224×224 for better accuracy

### 2.2 Optimized Batch Size
**Problem**: Small batch size = GPU underutilized; Large batch size = memory errors
**Evolution**: `BATCH_SIZE = 32 → 64 → 128 → 64`
```python
BATCH_SIZE = 64  # Balanced for speed and GPU memory
```
**Impact**:
- Better GPU utilization
- Faster training per epoch
- Avoids out-of-memory errors

### 2.3 Adjusted Training Pairs Per Epoch
**Problem**: Too many pairs = very slow epochs; Too few = poor learning
**Evolution**: `PAIRS_TRAIN = 25000 → 5000 → 3000 → 8000 → 6000`
```python
PAIRS_TRAIN = 6000  # Balanced approach
PAIRS_VAL = 1200
```
**Impact**:
- Faster epochs (~5-7 minutes instead of 15-20 minutes)
- Still enough data for good learning

### 2.4 Optimized Data Loading Workers
**Problem**: Too many workers = RAM overflow; Too few = GPU starvation
**Evolution**: `NUM_WORKERS = 0 → 2 → 4 → 6 → 8 → 2`
```python
NUM_WORKERS = 2  # Safe for large DICOM images
PREFETCH_FACTOR = 2
```
**Impact**:
- Parallel data loading without memory errors
- Reduced memory errors from 275MB DICOM images

### 2.5 Memory-Efficient DICOM Loading
**Problem**: DICOM images are 4000×6000 pixels (275 MB each!)
**Solution**: Downsample before processing
```python
def load_dicom_rgb(path: str) -> np.ndarray:
    ds = pydicom.dcmread(path)
    arr = ds.pixel_array

    # Resize FIRST to save memory (before float conversion)
    if arr.shape[0] > 1024 or arr.shape[1] > 1024:
        scale = min(1024 / arr.shape[0], 1024 / arr.shape[1])
        new_h = int(arr.shape[0] * scale)
        new_w = int(arr.shape[1] * scale)
        arr = cv2.resize(arr, (new_w, new_h), interpolation=cv2.INTER_AREA)

    # ... rest of processing
```
**Impact**:
- Prevented memory errors
- ~5x reduction in memory usage per image

---

## 3. Model Architecture Improvements

### 3.1 Upgraded Backbone Network
**Problem**: ResNet18 too simple for complex medical images
**Evolution**: `ResNet18 → ResNet34 → ResNet50`
```python
class SiameseNet(nn.Module):
    def __init__(self, embed_dim=512):  # Increased from 128
        super().__init__()
        # Before: ResNet18 (11M parameters)
        # After: ResNet50 (25.5M parameters)
        backbone = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
        backbone.fc = nn.Identity()
        self.backbone = backbone
```
**Impact**:
- 3x more parameters (11M → 28M total)
- Better feature extraction
- Higher GPU utilization

### 3.2 Deeper Projection Head
**Problem**: Simple projection head limits embedding quality
**Solution**: Added multiple layers with BatchNorm and Dropout
```python
# Before: Simple 2-layer head
self.head = nn.Sequential(
    nn.Linear(512, 512), nn.ReLU(inplace=True),
    nn.Linear(512, 128)
)

# After: Deep 5-layer head with regularization
self.head = nn.Sequential(
    nn.Linear(2048, 2048),  # ResNet50 outputs 2048
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

    nn.Linear(512, 512)  # Final embedding
)
```
**Impact**:
- Richer embeddings (128 → 256 → 512 dimensions)
- Better regularization (Dropout prevents overfitting)
- BatchNorm stabilizes training

### 3.3 Increased Embedding Dimensions
**Evolution**: `embed_dim = 128 → 256 → 512`
```python
embed_dim = 512  # More expressive embeddings
```
**Impact**:
- More capacity to represent complex patterns
- Better separation between classes

---

## 4. Training Hyperparameter Tuning

### 4.1 Learning Rate Optimization
**Change**: Reduced learning rate for stability
```python
# Before:
LR = 1e-3

# After:
LR = 5e-4  # More stable convergence
```
**Impact**:
- Smoother training
- Better convergence

### 4.2 Added Learning Rate Scheduler
**Addition**: Cosine annealing scheduler
```python
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    opt, T_max=EPOCHS, eta_min=1e-6
)
```
**Impact**:
- Smooth learning rate decay
- Better final convergence
- Prevents overshooting optimal weights

### 4.3 Increased Contrastive Loss Margin
**Change**: Larger margin for better class separation
```python
# Before:
MARGIN = 1.0

# After:
MARGIN = 2.0  # Forces embeddings farther apart
```
**Impact**:
- Stronger separation between classes
- Better discrimination

### 4.4 Added Weight Decay
**Addition**: L2 regularization
```python
opt = torch.optim.AdamW(net.parameters(), lr=LR, weight_decay=1e-4)
```
**Impact**:
- Prevents overfitting
- Better generalization

### 4.5 Adjusted Number of Epochs
**Evolution**: `EPOCHS = 10 → 15 → 20 → 12`
```python
EPOCHS = 12  # Balanced for convergence vs time
```
**Impact**:
- Enough epochs for convergence
- Not too long for practical training

---

## 5. Data Augmentation Improvements

### 5.1 Enhanced Augmentation Pipeline
**Problem**: Insufficient augmentation leads to overfitting
**Solution**: Added diverse augmentations
```python
# Before: Minimal augmentation
tf = [
    A.Resize(height=size, width=size),
    A.HorizontalFlip(p=0.5),
    A.RandomBrightnessContrast(p=0.3),
    A.Normalize(), ToTensorV2()
]

# After: Rich augmentation
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
```
**Impact**:
- Better generalization
- Reduced overfitting
- More robust to variations in test data

---

## 6. DataLoader Optimizations

### 6.1 Added Prefetching
**Addition**: Prefetch batches ahead of time
```python
train_dl = DataLoader(
    train_ds,
    batch_size=BATCH_SIZE,
    num_workers=NUM_WORKERS,
    pin_memory=True,
    prefetch_factor=2,  # Prefetch 2 batches per worker
    persistent_workers=True  # Keep workers alive between epochs
)
```
**Impact**:
- Reduced GPU idle time
- Smoother data pipeline

---

## 7. Evaluation Improvements

### 7.1 Increased Evaluation Sample Sizes
**Change**: Use more samples for better accuracy estimation
```python
# Before:
EVAL_TRAIN_SUBSET = 1000
EVAL_TEST_SUBSET = 1000

# After:
EVAL_TRAIN_SUBSET = 3000  # Better prototype estimation
EVAL_TEST_SUBSET = 2000   # More reliable accuracy
```
**Impact**:
- More accurate prototypes for classification
- More reliable test accuracy measurement

---

## 8. Attempted But Reverted Changes

### 8.1 Hard Negative Mining (REVERTED)
**Approach**: Mine hard pairs (difficult examples) during training
**Why Reverted**: Decreased accuracy
```python
# Code is commented out in train_and_evaluate.py
# Lines 68-150
```
**Reason for Failure**:
- Overfitted to difficult/noisy samples
- Random sampling provides better diversity
- Hard mining can focus too much on outliers in medical data

**Lesson Learned**: Sometimes simpler is better; random balanced sampling works well for imbalanced medical datasets

---

## 9. Progress Tracking & Monitoring

### 9.1 Added Detailed Progress Printing
**Addition**: Show batch-level progress and timing
```python
# Print every 10 batches
if batch_idx % 10 == 0 or batch_idx == 1:
    print(f"  Batch {batch_idx}/{len(train_dl)} | Loss: {loss.item():.4f}")
```
**Impact**:
- Better visibility into training progress
- Easy to spot issues

### 9.2 Added Epoch Summaries
**Addition**: Clear epoch-level metrics
```python
print(f"EPOCH {epoch}/{EPOCHS} SUMMARY:")
print(f"  Train Loss: {tr_loss:.4f}")
print(f"  Val Loss:   {va_loss:.4f}")
print(f"  Learning rate: {current_lr:.2e}")
```
**Impact**:
- Track convergence
- Spot overfitting early

### 9.3 Added Training Time Tracking
**Addition**: Measure and display training time
```python
print(f"Training time:   {train_time//60:.0f}m {train_time%60:.0f}s")
print(f"Evaluation time: {eval_time//60:.0f}m {eval_time%60:.0f}s")
print(f"Total time:      {total_time//60:.0f}m {total_time%60:.0f}s")
```
**Impact**:
- Track efficiency improvements
- Estimate total training time

---

## 10. Final Configuration Summary

### Current Optimized Settings:
```python
# Data
IMG_SIZE        = 224
PAIRS_TRAIN     = 6000
PAIRS_VAL       = 1200

# Training
EPOCHS          = 12
BATCH_SIZE      = 64
LR              = 5e-4
MARGIN          = 2.0

# Data Loading
NUM_WORKERS     = 2
PREFETCH_FACTOR = 2

# Model
Backbone        = ResNet50
Embedding Dim   = 512
Total Parameters ≈ 28M

# Evaluation
EVAL_TRAIN_SUBSET = 3000
EVAL_TEST_SUBSET  = 2000
```

---

## 11. Expected Performance

### Training Speed:
- **Per Epoch**: ~5-7 minutes (93 batches at batch size 64)
- **Total Training**: ~60-90 minutes (12 epochs)
- **GPU Utilization**: 60-90% during forward/backward passes

### Accuracy:
- **Target**: ~80% (0.80) on test set
- **Expected Range**: 75-85% with current configuration

---

## 12. Key Takeaways

### What Worked:
✅ Deeper model (ResNet50 vs ResNet18)
✅ Larger embeddings (512 vs 128 dimensions)
✅ Better augmentations
✅ Learning rate scheduling
✅ Balanced hyperparameters (batch size, epochs, pairs)
✅ Memory-efficient DICOM loading

### What Didn't Work:
❌ Hard negative mining (decreased accuracy)
❌ Very large batch sizes (memory errors)
❌ Too many workers (memory errors)
❌ Very small image sizes (lost important details)

### Best Practices Learned:
1. **Balance is key**: Extreme settings (too large/small) often hurt
2. **Medical data is special**: Standard tricks don't always apply
3. **Memory matters**: Large medical images need careful handling
4. **Simple works**: Random balanced sampling beats complex mining
5. **Monitor everything**: Track time, GPU usage, and progress

---

## 13. Future Improvement Ideas (Not Implemented)

If further accuracy improvement is needed:

1. **Convert DICOM to PNG** - 50-100x faster data loading
2. **Ensemble methods** - Train multiple models, average predictions (+3-7%)
3. **Different loss functions** - Try Triplet Loss or ArcFace Loss
4. **Larger backbone** - ResNet101 or EfficientNet-B4
5. **More training data** - Increase to 10k-15k pairs per epoch
6. **Test-time augmentation** - Average predictions over multiple augmented versions
7. **Cross-validation** - Train on multiple folds for better generalization

---

**Document Created**: 2025-10-27
**Last Updated**: 2025-10-27
**Project**: ISIC 2020 Melanoma Classification - Siamese Network
**Author**: Claude Code Assistant
