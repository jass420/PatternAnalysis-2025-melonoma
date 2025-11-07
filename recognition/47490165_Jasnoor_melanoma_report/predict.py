"""
predict.py - Example Usage of Trained Siamese Network

This script demonstrates how to use the trained model for inference.
Shows example predictions with visualizations.
"""

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from pathlib import Path
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, confusion_matrix
import cv2

# Import from custom modules
from modules import SiameseNet, PrototypeClassifier
from dataset import ImageDataset, load_dicom_rgb
import albumentations as A
from albumentations.pytorch import ToTensorV2

# ============== CONFIGURATION ==============
BASE = Path(r"C:\Users\mjas0\OneDrive\Desktop\courses\COMP3710\Alzheimer-s")
CATALOG_CSV = BASE / "train_mapping.csv"
MODEL_PATH = BASE / "outputs" / "best_siamese.pth"
OUT_DIR = BASE / "outputs"

IMG_SIZE = 224
BATCH_SIZE = 64
NUM_WORKERS = 2
SEED = 42
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

print("="*70)
print("SIAMESE NETWORK - PREDICTION EXAMPLES")
print("="*70)
print(f"Device: {DEVICE}")
print(f"Model: {MODEL_PATH}")
print()


def load_trained_model(model_path, device):
    """Load the trained Siamese network."""
    model = SiameseNet(embed_dim=512).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    return model


@torch.no_grad()
def compute_embeddings(model, dataloader, device):
    """Compute embeddings for images."""
    model.eval()
    all_embeds = []
    all_labels = []

    for x, y in dataloader:
        x = x.to(device)
        z = model(x).cpu().numpy()
        all_embeds.append(z)
        all_labels.append(y.numpy())

    return np.vstack(all_embeds), np.concatenate(all_labels)


def visualize_predictions(df, indices, predictions, true_labels, save_path):
    """
    Visualize sample predictions with images.

    Shows a grid of images with their true labels and predictions.
    """
    n_samples = len(indices)
    cols = 4
    rows = (n_samples + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(16, 4 * rows))
    axes = axes.flatten() if n_samples > 1 else [axes]

    class_names = {0: 'Normal', 1: 'Melanoma'}

    for idx, ax in enumerate(axes):
        if idx < n_samples:
            # Load and display image
            row = df.iloc[indices[idx]]
            img = load_dicom_rgb(row.dcm_path)

            # Resize for display
            img_display = cv2.resize(img, (224, 224))

            # Get prediction and true label
            pred = predictions[idx]
            true = true_labels[idx]

            # Display image
            ax.imshow(img_display)

            # Set title with prediction and true label
            color = 'green' if pred == true else 'red'
            title = f"True: {class_names[true]}\nPred: {class_names[pred]}"
            ax.set_title(title, fontsize=10, fontweight='bold', color=color)
            ax.axis('off')
        else:
            ax.axis('off')

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Prediction visualization saved to: {save_path}")
    plt.close()


def visualize_embedding_space(embeddings, labels, save_path):
    """
    Visualize embeddings in 2D using PCA.

    Shows how the model separates the two classes in embedding space.
    """
    from sklearn.decomposition import PCA

    # Reduce to 2D using PCA
    pca = PCA(n_components=2)
    embeddings_2d = pca.fit_transform(embeddings)

    # Plot
    fig, ax = plt.subplots(figsize=(10, 8))

    # Separate by class
    normal_mask = labels == 0
    melanoma_mask = labels == 1

    ax.scatter(embeddings_2d[normal_mask, 0], embeddings_2d[normal_mask, 1],
              c='blue', label='Normal', alpha=0.6, s=50)
    ax.scatter(embeddings_2d[melanoma_mask, 0], embeddings_2d[melanoma_mask, 1],
              c='red', label='Melanoma', alpha=0.6, s=50)

    ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)', fontsize=12)
    ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)', fontsize=12)
    ax.set_title('Embedding Space Visualization (PCA)', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Embedding space visualization saved to: {save_path}")
    plt.close()


def predict_single_image(model, image_path, train_embeds, train_labels, device):
    """
    Predict class for a single image.

    Args:
        model: Trained Siamese network
        image_path: Path to DICOM image
        train_embeds: Training embeddings for prototype computation
        train_labels: Training labels
        device: torch device

    Returns:
        tuple: (predicted_class, probabilities)
    """
    # Load and preprocess image
    img = load_dicom_rgb(image_path)

    transform = A.Compose([
        A.Resize(height=IMG_SIZE, width=IMG_SIZE),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ])

    x = transform(image=img)["image"].unsqueeze(0).to(device)

    # Compute embedding
    with torch.no_grad():
        embedding = model(x).cpu().numpy()

    # Classify using prototype
    classifier = PrototypeClassifier()
    classifier.fit(train_embeds, train_labels)

    prediction = classifier.predict(embedding)[0]
    probabilities = classifier.predict_proba(embedding)[0]

    return prediction, probabilities


def main():
    """Main prediction pipeline."""

    # Load model
    print("Loading trained model...")
    model = load_trained_model(MODEL_PATH, DEVICE)
    print("Model loaded successfully!")
    print()

    # Load dataset
    print("Loading dataset...")
    df = pd.read_csv(CATALOG_CSV)

    # Use a subset for demonstration
    np.random.seed(SEED)
    sample_indices = np.random.choice(len(df), size=500, replace=False)
    sample_df = df.iloc[sample_indices].reset_index(drop=True)

    print(f"Loaded {len(sample_df)} samples for prediction")
    print()

    # Create dataset and dataloader
    dataset = ImageDataset(sample_df, size=IMG_SIZE)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False,
                           num_workers=NUM_WORKERS, pin_memory=True)

    # Compute embeddings
    print("Computing embeddings...")
    embeddings, labels = compute_embeddings(model, dataloader, DEVICE)
    print(f"Computed embeddings: shape {embeddings.shape}")
    print()

    # Split into train/test for prototype-based classification
    n_train = int(0.7 * len(embeddings))
    train_embeds = embeddings[:n_train]
    train_labels = labels[:n_train]
    test_embeds = embeddings[n_train:]
    test_labels = labels[n_train:]

    # Classify using prototypes
    print("Making predictions...")
    classifier = PrototypeClassifier()
    classifier.fit(train_embeds, train_labels)
    predictions = classifier.predict(test_embeds)
    probabilities = classifier.predict_proba(test_embeds)

    # Compute accuracy
    accuracy = accuracy_score(test_labels, predictions)
    cm = confusion_matrix(test_labels, predictions)

    print()
    print("="*70)
    print("PREDICTION RESULTS")
    print("="*70)
    print(f"\nAccuracy: {accuracy:.4f} ({accuracy * 100:.2f}%)")
    print()
    print("Confusion Matrix:")
    print(cm)
    print("  [Normal→Normal, Normal→Melanoma]")
    print("  [Melanoma→Normal, Melanoma→Melanoma]")
    print()

    # Example predictions
    print("="*70)
    print("EXAMPLE PREDICTIONS")
    print("="*70)
    print()

    for i in range(min(5, len(test_labels))):
        pred = predictions[i]
        true = test_labels[i]
        probs = probabilities[i]

        class_names = {0: 'Normal', 1: 'Melanoma'}
        status = "✓ CORRECT" if pred == true else "✗ WRONG"

        print(f"Sample {i+1}:")
        print(f"  True Label:  {class_names[true]}")
        print(f"  Prediction:  {class_names[pred]}")
        print(f"  Confidence:  Normal={probs[0]:.3f}, Melanoma={probs[1]:.3f}")
        print(f"  Status:      {status}")
        print()

    # Visualizations
    print("="*70)
    print("GENERATING VISUALIZATIONS")
    print("="*70)
    print()

    # Visualize some predictions
    print("Creating prediction visualization...")
    test_df = sample_df.iloc[n_train:].reset_index(drop=True)

    # Select diverse examples (some correct, some wrong)
    correct_indices = np.where(predictions == test_labels)[0]
    wrong_indices = np.where(predictions != test_labels)[0]

    if len(correct_indices) >= 6 and len(wrong_indices) >= 2:
        selected = list(correct_indices[:6]) + list(wrong_indices[:2])
    else:
        selected = list(range(min(8, len(predictions))))

    visualize_predictions(
        test_df,
        selected,
        predictions[selected],
        test_labels[selected],
        OUT_DIR / "prediction_examples.png"
    )

    # Visualize embedding space
    print("Creating embedding space visualization...")
    visualize_embedding_space(
        embeddings,
        labels,
        OUT_DIR / "embedding_space.png"
    )

    print()
    print("="*70)
    print("SINGLE IMAGE PREDICTION EXAMPLE")
    print("="*70)
    print()

    # Example: predict a single image
    example_idx = 0
    example_row = df.iloc[example_idx]
    example_path = example_row.dcm_path

    print(f"Predicting single image: {example_row.image_name}")
    pred_class, pred_probs = predict_single_image(
        model, example_path, train_embeds, train_labels, DEVICE
    )

    class_names = {0: 'Normal', 1: 'Melanoma'}
    print(f"  Predicted Class: {class_names[pred_class]}")
    print(f"  Probabilities: Normal={pred_probs[0]:.3f}, Melanoma={pred_probs[1]:.3f}")
    print()

    print("="*70)
    print("PREDICTION COMPLETE")
    print("="*70)


if __name__ == "__main__":
    main()
