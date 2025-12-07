# DEEPFAKE IMAGE DETECTION USING EFFICIENTNET-B7

This project implements a binary deepfake vs. real image classifier using EfficientNet-B7, trained on FaceForensics++ extracted frames. The framework uses PyTorch, torchvision, and a clean, reproducible training pipeline.

## 1. PROJECT OVERVIEW
This repository provides a full training and evaluation framework for detecting deepfake images using:
- Framework: PyTorch - An open soure deep learning framework that provides python tools for building, training, and deploying neural networks.
- Problem: binary deepfake image classification (Original = 0, Deepfake = 1)
- Approach: Fine-tuning EfficientNet-B7
- Pretrained weights: ImageNet-1K
- Dataset: FF++ extracted frames dataset
- Custom PyTorch Dataset + Transforms + Dataloader
- Metrics: Accuracy, Precision, Recall, F1-Score, AUROC
- Visualization: Confusion Matrix, ROC Curve
    

## 2. DEPENDENCIES & LIBRARIES
**Core Libraries:**
- Python 3.8+
- PyTorch
- Torchvision
- Numpy
- Pandas
- Scikit-Learn
- Matplotlib


## 3. DATASET
**Dataset download Link:** https://www.kaggle.com/datasets/fatimahirshad/faceforensics-extracted-dataset-c23

**The dataset from the FaceForensics++ dataset consists of six categories:**
FF++C32-Frames:
- Original
- Deepfakes
- Face2Face
- FaceShifter
- FaceSwap
- NeuralTextures

Total Numbers of Classes : 6

For this project, 2 classes were used:
- Original (5000 images)
- Deepfakes (5000 images)


### Preprocessing Steps
**Custom Dataset:**
The custom dataset builds a combined dataframe of original and deepfake data:
    - Adds absolute file paths
    - Assigns class labels (Original = 0, Deepfakes = 1)
    - Produces a structured dataset for training

**Transforms:**
Images are resized to 224x224, normalized to ImageNet standards
    - Resize
    - RandomHorizontalFlip
    - RandomRotation
    - ColorJitter
    - Normalize


## 4. FRAMEWORK ARCHITECTURE
### Model:
    **EfficientNet-B7 pretrained on ImageNet**

    **The convolutional backbone:**
    This contains an initial 3x3 conv stem and 8 blocks (0–7), each made up of MBConv layers:
        **Each MBConv block contains:**
            - Conv2d (1×1 expansion conv)
            - BatchNorm2d
            - SiLU / Swish activation
            - Depthwise Conv2d
            - BatchNorm2d
            - Squeeze-and-Excitation (SE) module - SE reduces channels
            - Fully connected layers inside SE
            - Conv2d (1×1 projection conv)
            - BatchNorm2d

    **The classifier:**
        - Modified classifier (Replaced classifier head with a binary classifier)

    
    **Original Pretraining Dataset:**
    - The model is pretrained on **ImageNet-1K**, which contains:
    - 1.2 million images
    - 1000 classes

### Loss Function:
    - BCEWithLogitsLoss

### Optimizer:
    - Adam optimizer



### Pipeline:
The script performs:
    - Seed setting for reproducibility
    - Train/Val/Test Split (70%/15%/15%)
    - DataLoader with workers + pinned memory
    - EfficientNet-B7 fine-tuning (which part does this)
    - Epoch-level metrics
    - Early stopping
    - Checkpoint saving best model

## 5. EVALUATION METRICS
The following were computed on the test set:
    - **Accuracy:** overall correctness 
    - **Precision:** how often fake predictions are correct
    - **Recall:** how many fakes we successfully detect
    - **F1 Score:** harmonic balance of precision and recall
    - **AUROC:** how well the model separates real from fake across thresholds 
    - **Confusion Matrix:** shows the counts of true/false positives and negatives
    - **Classfication Report:** A detailed summary of precision, recall, F1-score, and support for each class.
    - **ROC Curve Visualization:** shows how the true positive rate varies against the false positive rate.


## 6. FINAL RESULTS

        | Metric     | Value   |
        |------------|--------|
        | Accuracy   | 0.9807 |
        | Precision  | 0.9814 |
        | Recall     | 0.9801 |
        | F1 Score   | 0.9808 |
        | AUROC      | 0.9966 |


    Plots generated:
    - Confusion matrix heatmap
    - ROC Curve

    Best epoch: 91
    Early stopping triggered: Yes
    Model Checkpoint: After training, the best model is saved as: best_model_efficientnet_b7.pth
    Reproducibility: The script includes explicit seed settings with CUDNN set to deterministic for reproducible runs.

## 7. INTERPRETATION OF RESULTS 
These metrics strongly suggest that:
    - EfficientNet-B7 is highly effective for deepfake image detection.
    - The model achieved strong discriminative performance demonstrating its ability to reliably separate real and fake images.
    - AUROC close to 1.0 suggests the model is highly confident in its predictions.
    - Misclassifications are minimal and evenly distributed, showing robust generalization.


## 8. REFERENCES

EfficientNet
Tan, M. and Le, Q.V. (2019) EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks. Proceedings of the 36th International Conference on Machine Learning, ICML 2019, Long Beach, 9-15 June 2019, 6105-6114. https://proceedings.mlr.press/v97/tan19a/tan19a.pdf

FaceForensics++ Dataset
Rössler, A., Cozzolino, D., Verdoliva, L., Riess, C., Thies, J., & Nießner, M. (2019). FaceForensics++: Learning to detect manipulated facial images. In Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV) (pp. 1–11).