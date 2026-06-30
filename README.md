# Deepfake Image Detection using EfficientNet-B7

This project implements a deep learning-based binary image classifier for detecting **deepfake** and **real** images using **EfficientNet-B7** and **transfer learning**. The model was fine-tuned on the **FaceForensics++** dataset using **PyTorch** and achieved strong performance on unseen test data.

The project focuses on building a reproducible deepfake detection pipeline covering:
- Data preprocessing and augmentation
- Transfer learning with EfficientNet-B7
- Model training and evaluation
- Performance visualization and analysis

---

## Key Features

- Binary deepfake vs real image classification
- EfficientNet-B7 fine-tuning with ImageNet pretrained weights
- PyTorch-based training pipeline
- Evaluation using Accuracy, Precision, Recall, F1-Score, and AUROC
- Confusion Matrix and ROC Curve visualization
- Reproducible training with deterministic seed configuration
- Early stopping and checkpoint saving

---

## Technologies Used

- Python
- PyTorch
- Torchvision
- Scikit-Learn
- Pandas
- NumPy
- Matplotlib

---

## Dataset

The model was trained on extracted frames from the **FaceForensics++** dataset using:
- Original images
- Deepfake manipulated images

The dataset was preprocessed using:
- Resize
- Normalize
- RandomHorizontalFlip
- RandomRotation
- ColorJitter

These preprocessing and augmentation techniques help improve model generalization.

---

## Model Architecture

The project uses **EfficientNet-B7**, a convolutional neural network architecture optimized for high accuracy and computational efficiency through compound scaling.

Transfer learning was applied by:
- Using ImageNet pretrained weights
- Replacing the original classifier head with a binary classifier
- Fine-tuning the network on deepfake image data

---

## Results

The model achieved strong performance on the test dataset:

| Metric | Score |
|---|---|
| Accuracy | 98.07% |
| Precision | 98.14% |
| Recall | 98.01% |
| F1 Score | 98.08% |
| AUROC | 99.66% |

The results demonstrate the effectiveness of EfficientNet-B7 for deepfake image classification while maintaining balanced precision and recall performance.

---

## Research Focus

This project also explores challenges in real-world deepfake detection, including:
- Generalization to unseen deepfake techniques
- Distribution shift and dataset bias
- Model explainability
- Robustness against evolving synthetic media generation methods

Potential future improvements include:
- Multimodal detection
- Continual learning approaches
- Explainable AI (XAI)
- Domain adaptation techniques
