
MODEL: EfficientNet-B7 (Pretrained on ImageNet-1K)

MODEL ARCHITECTURE DOCUMENTATION:
EfficientNet-B7 is a convolutional neural network that uses:
# - Mobile Inverted Bottleneck Convolution (MBConv) layers
# - Depthwise separable convolutions
# - Swish activation function (SiLU in PyTorch)
# - Squeeze-and-Excitation (SE) blocks - FC layer: Computes channel attention. Learns which channels to emphasize or suppress.
#
# ORIGINAL PRE-TRAINING DATASET:
# - The model is pretrained on **ImageNet-1K**, which contains:
#   - 1.2 million images
#   - 1000 classes
#
# ACTIVATION FUNCTIONS:
# - All convolution blocks use **SiLU (Swish)** activations.

# Replace the classifier head:
        # Original: Linear(2560 -> 1000)
        # New:     Linear(2560 -> 1)     for binary classification
#
# LOSS FUNCTION:
# - For binary deepfake classification, we use:
#   BCEWithLogitsLoss()
#   Because it combines:
#     - Sigmoid activation
#     - Binary cross entropy loss

# Load pretrained EfficientNet-B7 weights (ImageNet-1K)

DETAILS:
1. Convolutional Layers

EfficientNet-B7 is composed of multiple MBConv (Mobile Inverted Bottleneck Convolution) blocks. 
These blocks include:
    - Depthwise convolution: Performs spatial filtering channel-by-channel, reducing computation.
    - Pointwise convolution (1×1): Expands or projects channel dimensions.
    - Inverted bottleneck structure: Expands channels first, processes them, then compresses them.
    - Squeeze-and-Excitation (SE) attention modules: Adaptively recalibrate feature maps based on global

Block Types Used- EfficientNet-B7 uses:
    MBConv1	1×	3×3	Early layers
    MBConv6	6×	3×3 / 5×5	Majority of network
    Depthwise Separable Convolution	—	3×3 / 5×5	Inside each MBConv block

In summary, the backbone is a deep stack of inverted residual convolution blocks with channel-wise attention.

