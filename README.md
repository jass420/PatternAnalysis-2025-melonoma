# COMP3710 Molonaoma detection

**Problem Description** /n
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
