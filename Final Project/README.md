# DEEPFAKE IMAGE DETECTION USING EFFICIENTNET-B7

This project implements a binary deepfake vs. real image classifier using EfficientNet-B7, finetuned on FaceForensics++ extracted frames. The framework uses PyTorch, torchvision, and a clean, reproducible training pipeline.

## 1. PROJECT OVERVIEW
This repository provides a full training and evaluation framework for detecting deepfake images using:
- **Framework:** PyTorch - An open soure deep learning framework that provides python tools for building, training, and deploying neural networks.
- **Problem:** Binary deepfake image classification (Original = 0, Deepfake = 1)
- **Approach:** Fine-tuning EfficientNet-B7 and transfer learning
- **Pretrained weights:** ImageNet-1K
- **Dataset:** FF++ extracted frames dataset, Custom PyTorch Dataset
- **Evaluation with Metrics:** Accuracy, Precision, Recall, F1-Score, AUROC
- **Visualization:** Confusion Matrix, ROC Curve
    

## 2. DEPENDENCIES & LIBRARIES
**Core Libraries:**
- PyTorch
- Torchvision
- Numpy
- Pandas
- Scikit-Learn
- Matplotlib
- Random 

**DESCRIPTION**
- PyTorch – A deep learning framework for building and training neural networks.
- Torchvision – A PyTorch library providing datasets, models, and image transformations.
- NumPy – A library for fast numerical computations and array manipulations.
- Pandas – A library for data manipulation and analysis (DataFrames).
- Scikit-Learn – A library for machine learning algorithms, data preprocessing, and evaluation metrics.
- Matplotlib – A library for creating plots and visualizations in Python.
- Random – A built-in Python module for generating random numbers and selections.

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

**For this project, 2 classes were used:**
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
### EfficientNet-B7 pretrained on ImageNet
- Increases efficiency of CNNs by systematically scaling the model's architecture and parameters.
- Uniformly scale depth, width, and resolution using compound coefficient (Compound Scaling)
- Computationally efficient

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

**Brief Descriptipn**

- Conv2d (1×1 expansion conv) - Expands the number of channels to a higher dimension before applying depthwise convolution
- BatchNorm2d - Normalizes feature maps across the batch to stabilize and speed up training.
- SiLU / Swish activation - Applies a smooth nonlinear activation that improves gradient flow and model performance
- Depthwise Conv2d - Applies a separate convolution per channel, drastically reducing computational cost.
- BatchNorm2d - Re-normalizes the depthwise convolution output to maintain stable activations.
- Squeeze-and-Excitation (SE) module - SE reduces channels - Learns channel-wise attention weights by squeezing spatial information and      reweighting channels.
- Fully connected layers inside SE - Compress and then expand channel information to compute attention weights.
- Conv2d (1×1 projection conv) - Reduces the expanded channels back to the desired output dimension.
- BatchNorm2d - Normalizes the final projected features to stabilize output

**The classifier:**
- Initial classifier (predicts 1000 classes)
- Modified classifier (Replace classifier head with a binary classifier)

    
**Original Pretraining Dataset:**
- The model is pretrained on **ImageNet-1K**, which contains:
- 1.2 million images
- 1000 classes

**Number of Parameters:**
- 66347960 (66M+)

### Loss Function:
- Original Loss function is **Cross Entropy Loss** (inferred)
- Loss Function Used is **BCEWithLogitsLoss**

### Optimizer:
- Adam optimizer

**Model training technique:**
- Transfer learning is applied by using a pre-trained model as a feature extractor, with a new classification head (linear layer) added on top.
- During training, the weights of both the backbone (feature extractor) and the classification head are updated.
- This allows the model to adapt to the new task and be fine-tuned on the specific dataset for deepfake detection.

### Pipeline:
**The script performs:**

- Seed setting for reproducibility
- Train/Val/Test Split (70%/15%/15%)
- DataLoader with workers + pinned memory
- EfficientNet-B7 fine-tuning
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

**Other metrics that might have been appropriate:**
- **Log Loss Metrics** quantifies the divergence between predicted probabilities and actual labels.

## 6. FINAL RESULTS

**Best epoch:** 91
**Best training loss and accuracy:**   Train Loss: 0.0059, Acc: 0.9987
**Best validation loss and accuracy:** Val Loss: 0.0452, Acc: 0.9893
**Early stopping triggered:** Yes
**Model Checkpoint:** After training, the best model is saved as: best_model_efficientnet_b7.pth
**Reproducibility:** The script includes explicit seed settings with CUDNN set to deterministic for reproducible runs.

**Result on Test Set**

        | Metric     | Value  |
        |------------|--------|
        | Accuracy   | 0.9807 |
        | Precision  | 0.9814 |
        | Recall     | 0.9801 |
        | F1 Score   | 0.9808 |
        | AUROC      | 0.9966 |

**Plots generated:**
   - Confusion matrix heatmap
   - ROC Curve

## 7. INTERPRETATION OF RESULTS 
- The model performed optimally at epoch 91 according to the validation loss/accuracy.
- Training Loss: 0.0059, Accuracy: 0.9987 : Extremely low loss and very high accuracy on the training set indicate the model has learned the training data very well.
- Validation Loss: 0.0452, Accuracy: 0.9893 : Low validation loss and high accuracy indicate good generalization to unseen data during training.
  Slight difference between train and validation metrics suggests minimal overfitting, which is good.
- The training was stopped automatically when validation performance stopped improving, preventing overfitting.
- Accuracy shows that 98% of samples classified correctly.
- Precision indicates that of all predicted fakes, 98.14% were actually fake (low false positives).
- Recall reveals that of all actual fakes, 98.01% were correctly detected (low false negatives).
- AUROC close to 1.0 suggests the model is highly confident in its predictions.

**These metrics strongly suggest that:**
- The model shows high overall performance on unseen test data.
- Precision ≈ Recall indicates the model is balanced between false positives and false negatives.
- AUROC of approximately 1 shows that the model can distinguish real vs. fake very reliably across different thresholds.

- EfficientNet-B7 is highly effective for deepfake image detection.
- The model achieved strong discriminative performance demonstrating its ability to reliably separate real and fake images.
- Misclassifications are minimal and evenly distributed.


## 8. Challenges
- Degraded performance on **unseen or distribution-shifted datasets**, especially heavily augmented or modified versions (e.g., augmented DFDC).
- Reduced generalization when encountering **real-world deepfake variations** not present during training.

## 9. Proposed Solutions
- **Self-supervised learning strategies** to improve representation quality and robustness without relying solely on labeled data.
- **Knowledge distillation and domain adaptation** to enhance the model’s ability to classify unseen and shifted samples effectively.
- **Training on diverse datasets** combined with **advanced data augmentation techniques** to improve generalization across deepfake sources and manipulation styles.


## 10. REFERENCES

EfficientNet
Tan, M. and Le, Q.V. (2019) EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks. Proceedings of the 36th International Conference on Machine Learning, ICML 2019, Long Beach, 9-15 June 2019, 6105-6114. https://proceedings.mlr.press/v97/tan19a/tan19a.pdf

FaceForensics++ Dataset
Rössler, A., Cozzolino, D., Verdoliva, L., Riess, C., Thies, J., & Nießner, M. (2019). FaceForensics++: Learning to detect manipulated facial images. In Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV) (pp. 1–11).

Khan, A.A., Laghari, A.A., Inam, S.A. et al. A survey on multimedia-enabled deepfake detection: state-of-the-art tools and techniques, emerging trends, current challenges & limitations, and future directions. Discov Computing 28, 48 (2025). https://doi.org/10.1007/s10791-025-09550-0 

Khan, S. A., & Dang-Nguyen, D.-T. (2023). Deepfake detection: A comparative analysis. arXiv. https://arxiv.org/abs/2308.03471