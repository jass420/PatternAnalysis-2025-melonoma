# COMP3710 Molonaoma detection

**Problem Description** 
*Medical Context*
We're solving a binary classification problem for melanoma detection using the ISIC 2020 Kaggle Challenge dataset. The task is to distinguish between:
Class 0: Normal/Benign skin lesions (25,704 samples)
Class 1: Melanoma (malignant skin cancer) (457 samples)
Key Challenges
Severe Class Imbalance: 98.2% normal vs 1.8% melanoma
High-Resolution Medical Images: DICOM format, up to 4000×6000 pixels
Visual Similarity: Early melanoma can look very similar to benign moles
Target Accuracy: ~80% on test set
Siamese Network Algorithm
Architecture Overview
The Siamese network uses a twin neural network architecture with shared weights to learn similarity between image pairs.

**Algorithm description**

### Overview
The Siamese Network is a twin neural network architecture that learns to distinguish between melanoma and normal skin lesions by learning a similarity metric. Instead of directly predicting classes, it maps images into an embedding space where similar images (same class) are close together and dissimilar images (different classes) are far apart.

### Architecture
```
Image Pair:
    Image 1 (224×224×3)          Image 2 (224×224×3)
           ↓                              ↓
    ┌──────────────────┐          ┌──────────────────┐
    │  ResNet50 CNN    │          │  ResNet50 CNN    │
    │ (Shared Weights) │          │ (Shared Weights) │
    └──────────────────┘          └──────────────────┘
           ↓                              ↓
     Features (2048)                Features (2048)
           ↓                              ↓
    ┌──────────────────┐          ┌──────────────────┐
    │ Projection Head  │          │ Projection Head  │
    │  (5 MLP layers)  │          │  (5 MLP layers)  │
    └──────────────────┘          └──────────────────┘
           ↓                              ↓
    Embedding (512)                 Embedding (512)
           └──────────────┬──────────────┘
                         ↓
                  Euclidean Distance
                         ↓
                  Contrastive Loss
```

### Key Components

**1. Shared Encoder (ResNet50)**
- Pre-trained on ImageNet for general visual features
- Both images processed with identical weights
- Extracts 2048-dimensional feature vectors

**2. Projection Head**
Transforms features into embeddings:
```
2048 → 2048 (BatchNorm + ReLU + Dropout 0.3)
2048 → 1024 (BatchNorm + ReLU + Dropout 0.3)
1024 → 1024 (BatchNorm + ReLU + Dropout 0.2)
1024 → 512  (BatchNorm + ReLU + Dropout 0.2)
512  → 512  (L2 Normalize)
```

**3. Contrastive Loss**
```
Given pair (z₁, z₂) with label y:
  - y = 1 if same class
  - y = 0 if different classes

Distance: d = ||z₁ - z₂||₂

Loss = y · d² + (1-y) · max(0, margin - d)²

Objective:
  - Similar pairs: Minimize distance (d → 0)
  - Dissimilar pairs: Maximize distance (d > 2.0)
```

### Training Process

**Balanced Pair Sampling** (handles 98.2% class imbalance):
- 6,000 pairs per epoch:
  - 3,000 positive pairs (same class)
  - 3,000 negative pairs (different classes)

**Data Augmentation**:
- Horizontal/Vertical flips
- Rotation (±20°)
- Brightness/Contrast adjustment
- Color shifts, Gaussian noise

**Optimization**:
- Optimizer: AdamW (lr=5e-4, weight decay=1e-4)
- LR Scheduler: Cosine annealing
- Batch size: 64 pairs
- Epochs: 12

### Classification (Inference)

**Prototype-based Classification**:
1. Compute class prototypes (average embeddings):
   - prototype_normal = mean(normal_embeddings)
   - prototype_melanoma = mean(melanoma_embeddings)

2. For test image:
   - Compute embedding: z_test
   - Find distances to both prototypes
   - Predict: nearest prototype's class

### Why This Works

- **Handles imbalance**: Balanced pair sampling instead of class frequencies
- **Few-shot learning**: Works with only 457 melanoma samples
- **Metric learning**: Learns meaningful similarity measure
- **Transfer learning**: Leverages ImageNet pretrained features

### Model Specifications
- Parameters: ~28 million
- Embedding dimension: 512
- Training time: 600 minutes (15 epochs)
- Accuracy: 80% 


**Dependencies**

```bash
# Python 3.8+ required
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118  # For CUDA 11.8
pip install pydicom
pip install opencv-python
pip install albumentations
pip install scikit-learn
pip install pandas
pip install numpy
```

**Required Packages:**
- `torch >= 2.0.0` - PyTorch deep learning framework with CUDA support
- `torchvision >= 0.15.0` - Pre-trained models (ResNet50) and transforms
- `pydicom >= 2.3.0` - Reading DICOM medical image format
- `opencv-python >= 4.7.0` - Image processing and resizing
- `albumentations >= 1.3.0` - Advanced data augmentation pipeline
- `scikit-learn >= 1.2.0` - Train/test splitting and evaluation metrics
- `pandas >= 1.5.0` - Data handling and CSV operations
- `numpy >= 1.24.0` - Numerical operations and array processing

**Hardware Requirements:**
- GPU: NVIDIA GPU with 6GB+ VRAM (tested on RTX series)
- RAM: 16GB+ recommended (DICOM images are large)
- Storage: ~40GB for ISIC 2020 dataset

**Confusion Matrix + accuracy:**

After multiple fine tunings and changes to the code, the simese network achieved an accuracy of 80.20% after training for 606 minutes. 

Confusion Matrix:
[[1598  374]
 [  22    6]]
  [Normal→Normal, Normal→Melanoma]
  [Melanoma→Normal, Melanoma→Melanoma]

**Pre-processing**

The preprocessing pipeline in [train_and_evaluate.py](train_and_evaluate.py) prepares DICOM medical images for training:

**DICOM Loading**: Raw DICOM files are loaded using pydicom. Large images (4000×6000, ~275MB) are downsampled to max 1024×1024 before processing to prevent memory errors. Grayscale images are converted to RGB by stacking 3 channels. Pixel values are normalized to [0,1] range then converted to uint8 [0,255].

**Data Augmentation** (Training): Images are resized to 224×224 and augmented using Albumentations library with random horizontal/vertical flips (p=0.5), rotation (±20°, p=0.5), brightness/contrast adjustments (p=0.3), and Gaussian noise (p=0.2). ImageNet normalization is applied before converting to PyTorch tensors.

**Validation**: Minimal preprocessing with resize to 224×224, ImageNet normalization, and tensor conversion only.

**Pair Generation**: The Siamese network requires image pairs. Positive pairs (same class) and negative pairs (different classes) are generated with balanced sampling to handle the severe class imbalance (98.2% normal vs 1.8% melanoma). 6000 training pairs per epoch (random) and 1200 fixed validation pairs are used.

**Training, validation, split justification**

The dataset is split using GroupShuffleSplit (80% train, 20% validation) grouped by patient_id to prevent data leakage. Multiple iterative improvements were made to achieve the target accuracy:

**Progression (oldest → newest):**
1. **66.5%** - Initial baseline with ResNet18, 128-dim embeddings, minimal augmentation, IMG_SIZE=256
2. **68.6%** - Reduced IMG_SIZE to 128 for faster loading, increased batch size to 64
3. **48.8%** - Attempted hard negative mining (failed - decreased accuracy, reverted). This was done by writing a function which increased the pairing of images in the first class and the second class incstead of same class.
4. **77.5%** - Upgraded to ResNet50, increased embedding to 512-dim, deeper projection head with BatchNorm/Dropout, increased margin to 2.0
5. **45.85%** - Adjusted workers/pairs configuration. 
6. **80.20%** - Final optimized configuration: IMG_SIZE=224, LR=5e-4, cosine annealing scheduler, enhanced augmentations (vertical flip, rotation ±20°, HSV, GaussNoise), memory-efficient DICOM loading, 6000 training pairs, 12 epochs

**Key improvements:** ResNet50 backbone (3× more parameters), 512-dim embeddings (4× larger), deeper 5-layer projection head, larger margin (2.0), learning rate scheduler, richer augmentations, balanced pair sampling for class imbalance (98.2% normal vs 1.8% melanoma).

**Training time:** 606 minutes (15 epochs)

