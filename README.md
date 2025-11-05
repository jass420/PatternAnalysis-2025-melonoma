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
- Training time: 60-90 minutes (12 epochs)
- Expected accuracy: 75-85% 


**Dependencies**

**Confusion Matrix:**


Confusion Matrix:
[[1598  374]
 [  22    6]]
  [Normal→Normal, Normal→Melanoma]
  [Melanoma→Normal, Melanoma→Melanoma]

**Pre-processing**

**Training, validation, split justification**
Multiple changes were made to the training to improve the accuracy. Below is a list of all the accuracies:

Accuracies:
80.20%
45.85%
77.5%
48.8%
68.6%
66.5%

