# SnowPetrel

Pose Estimation and Behavior Classification of Snow Petrel Bird with Batch Inference.


## Pose Estimation

1. ResNet50
2. Conv Head
3. Output (Number of Keypoints)

## Classification

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