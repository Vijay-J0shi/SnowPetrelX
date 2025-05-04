# SnowPetrel

Welcome to the Snow Petrel Behavior Analysis System! This project empowers ecological researchers studying Indian wildlife to analyze snow petrel behavior using advanced deep learning techniques. It focuses on two core tasks:

- **Pose Estimation**: Detecting key body parts (head, beak, neck, body, tail) from images to understand the bird's posture.
- **Behavior Classification**: Identifying specific behaviors (nesting, preening) based on keypoint data to study ecological patterns.

## Project Overview

The system leverages deep learning models to process images and keypoint data, enabling precise analysis of snow petrel behavior. Its main components include:

- **Pose Estimation**: Uses a ResNet50-based model to predict the coordinates of key body parts in images.
- **Behavior Classification**: Employs a Graph Convolutional Network (GCN) for spatial analysis of keypoints and a Gated Recurrent Unit (GRU) for temporal analysis to classify behaviors.
- **Data Preprocessing**: Cleans and organizes the dataset, using spline interpolation to handle missing keypoint values.
- **Evaluation**: Assesses pose estimation with the Percentage of Correct Keypoints with Head Normalization (PCKh) metric and behavior classification with accuracy, precision, recall, and F1-score.
- **Visualization**: Generates visual plots of predicted keypoints overlaid on images for intuitive inspection.


## Definitions
- **PCKh**: Percentage of Correct Keypoints with Head Normalization
- **GCN**: Graph Convolutional Network
- **GRU**: Gated Recurrent Unit
- **ResNet50**: Residual Network with 50 layers
- **PyTorch**: Open-source machine learning framework
- **Pandas**: Data analysis library
- **NumPy**: Numerical computing library
- **Matplotlib**: Plotting library

## Pose estimation Model
</br>
<img src="https://github.com/user-attachments/assets/9e017bae-231e-4e07-bb32-efc1719767fd" width="400" height="300" />

</br>
Output is cordinates = head, top beak, bottom beak , neck, body, tail

```
Input: RGB Image [3, H, W] (e.g., [3, 224, 224])
    ↓
Image Preprocessing:
    - Resize to fixed dimensions (e.g., 224x224)
    - Normalize pixel values to [0, 1]
    - Convert to tensor
    ↓
ResNet50 Backbone:
    - Convolutional layers extract spatial features
    - Residual connections for deep feature learning
    - Outputs feature maps [C, H', W'] (e.g., [2048, 7, 7])
    ↓
Convolutional Head:
    - Additional conv layers for keypoint-specific features
    - Batch normalization for training stability
    - ReLU (BirdPoseModel) or SiLU (BirdPoseModelX) activation
    - Dropout (BirdPoseModelX) for regularization
    ↓
Output Layer:
    - Fully connected or conv layer to predict keypoint coordinates
    - Outputs [num_keypoints, 2] (x, y coordinates for each keypoint)
    ↓
Keypoint Normalization:
    - Denormalize coordinates to original image dimensions
    ↓
Output: Keypoint Coordinates [num_keypoints, 2] (e.g., [7, 2])
```

## Classification Model  Working 

> Behaviors: `nesting`, `preening`

```
Input: Keypoints [T, num_keypoints, 2] (e.g., [30, 8, 2])
    ↓
Graph Feature Extractor (GCN):
    - Models relationships between body parts
    - Outputs spatial embeddings [T, num_keypoints, d]
    ↓
Temporal Module (GRU ):
    - Captures temporal dynamics in keypoint movement
    - Outputs temporal embeddings [T, d]
    ↓
Global Average Pooling:
    - Aggregates information across time
    ↓
Fully Connected Layers:
    - Dense layers for classification
    - Dropout for regularization
    ↓
Output: Behavior Class Probabilities
```

