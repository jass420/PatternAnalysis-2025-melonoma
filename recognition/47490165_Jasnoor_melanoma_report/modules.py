"""
modules.py - Neural Network Components for Siamese Network

Contains:
- SiameseNet: The main Siamese network model with ResNet50 backbone
- ContrastiveLoss: Loss function for training Siamese networks
- PrototypeClassifier: Classifier using prototype-based inference
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
import numpy as np


class SiameseNet(nn.Module):
    """
    Siamese Network with ResNet50 backbone and deep projection head.

    Architecture:
    - Backbone: ResNet50 pretrained on ImageNet (extracts 2048-dim features)
    - Projection Head: 5-layer MLP with BatchNorm and Dropout (maps to embed_dim)
    - Output: L2-normalized embeddings for metric learning

    Args:
        embed_dim (int): Dimension of output embeddings (default: 512)
    """

    def __init__(self, embed_dim=512):
        super().__init__()

        # ResNet50 backbone (pretrained on ImageNet)
        try:
            backbone = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
        except Exception:
            backbone = models.resnet50(pretrained=True)

        # Remove final classification layer
        backbone.fc = nn.Identity()
        self.backbone = backbone

        # Deep projection head with regularization
        # Maps 2048-dim ResNet features to embed_dim embeddings
        self.head = nn.Sequential(
            nn.Linear(2048, 2048),
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
        """
        Forward pass through the network.

        Args:
            x (torch.Tensor): Input images of shape (batch_size, 3, 224, 224)

        Returns:
            torch.Tensor: L2-normalized embeddings of shape (batch_size, embed_dim)
        """
        # Extract features using ResNet50 backbone
        features = self.backbone(x)  # (batch_size, 2048)

        # Project to embedding space
        embeddings = self.head(features)  # (batch_size, embed_dim)

        # L2 normalization for stable metric learning
        return F.normalize(embeddings, dim=1)


class ContrastiveLoss(nn.Module):
    """
    Contrastive Loss for Siamese Networks.

    Loss formula:
    - Positive pairs (same class): L = distance^2
    - Negative pairs (different class): L = max(0, margin - distance)^2

    This encourages:
    - Similar images to have small distance
    - Dissimilar images to be at least 'margin' apart

    Args:
        margin (float): Minimum distance for negative pairs (default: 2.0)
    """

    def __init__(self, margin=2.0):
        super().__init__()
        self.margin = margin

    def forward(self, z1, z2, y):
        """
        Compute contrastive loss.

        Args:
            z1 (torch.Tensor): Embeddings from first image (batch_size, embed_dim)
            z2 (torch.Tensor): Embeddings from second image (batch_size, embed_dim)
            y (torch.Tensor): Labels (1.0 = same class, 0.0 = different class)

        Returns:
            torch.Tensor: Scalar loss value
        """
        # Euclidean distance between embeddings
        distance = F.pairwise_distance(z1, z2)

        # Loss for positive pairs (pull together)
        pos_loss = y * (distance ** 2)

        # Loss for negative pairs (push apart by margin)
        neg_loss = (1 - y) * (F.relu(self.margin - distance) ** 2)

        # Average loss
        return (pos_loss + neg_loss).mean()


class PrototypeClassifier:
    """
    Prototype-based classifier for Siamese network embeddings.

    Classification strategy:
    1. Compute class prototypes (mean embedding for each class) from training set
    2. Classify test samples by finding nearest prototype (Euclidean distance)

    This is a simple but effective approach for few-shot and imbalanced learning.
    """

    def __init__(self):
        self.prototypes = {}

    def fit(self, embeddings, labels):
        """
        Compute class prototypes from training embeddings.

        Args:
            embeddings (np.ndarray): Training embeddings (n_samples, embed_dim)
            labels (np.ndarray): Training labels (n_samples,)
        """
        unique_classes = np.unique(labels)
        self.prototypes = {}

        for cls in unique_classes:
            # Compute mean embedding for this class
            class_embeddings = embeddings[labels == cls]
            self.prototypes[cls] = class_embeddings.mean(axis=0)

        return self

    def predict(self, embeddings):
        """
        Predict class labels for test embeddings.

        Args:
            embeddings (np.ndarray): Test embeddings (n_samples, embed_dim)

        Returns:
            np.ndarray: Predicted class labels (n_samples,)
        """
        if not self.prototypes:
            raise ValueError("Classifier not fitted. Call fit() first.")

        predictions = []

        for emb in embeddings:
            # Compute distance to each class prototype
            distances = {
                cls: np.linalg.norm(emb - proto)
                for cls, proto in self.prototypes.items()
            }

            # Predict class with minimum distance
            pred_class = min(distances, key=distances.get)
            predictions.append(pred_class)

        return np.array(predictions)

    def predict_proba(self, embeddings):
        """
        Predict class probabilities using softmax over negative distances.

        Args:
            embeddings (np.ndarray): Test embeddings (n_samples, embed_dim)

        Returns:
            np.ndarray: Class probabilities (n_samples, n_classes)
        """
        if not self.prototypes:
            raise ValueError("Classifier not fitted. Call fit() first.")

        probabilities = []
        classes = sorted(self.prototypes.keys())

        for emb in embeddings:
            # Compute distances to all prototypes
            distances = np.array([
                np.linalg.norm(emb - self.prototypes[cls])
                for cls in classes
            ])

            # Convert distances to probabilities (softmax over negative distances)
            # Negative because smaller distance = higher probability
            probs = np.exp(-distances) / np.exp(-distances).sum()
            probabilities.append(probs)

        return np.array(probabilities)
