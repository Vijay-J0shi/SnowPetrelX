import os
from PIL import Image

from sklearn.metrics import precision_recall_fscore_support 

import pandas as pd
import torch

import torch.nn as nn
import torch_geometric.nn as gnn

from torch.utils.data import Dataset
from rich import print



class BehaviorDataset(Dataset):
    def __init__(
        self,
        dataframe: pd.DataFrame,
        dataset_root_folder: str,
        img_transform: callable = None,
        kp_transform: callable = None,
        time_steps: int = 1,
    ):
        """
        Dataset for behavior classification using image, keypoint data, and graph edges.

        Args:
            dataframe (pd.DataFrame): The DataFrame containing annotations.
            dataset_root_folder (str): Root folder where images are stored.
            img_transform (callable, optional): Transform for the images.
            kp_transform (callable, optional): Transform for the keypoints.
            time_steps (int): Number of consecutive frames to form a sequence.
        """
        self.annotations = dataframe
        self.dataset_root_folder = dataset_root_folder
        self.img_transform = img_transform
        self.kp_transform = kp_transform
        self.time_steps = time_steps

        # Dynamically map behavior labels to indices
        self.label_mapping = ["nesting", "preening"]

        # Define static edges for keypoints graph
        self.edges = torch.tensor([
            (0, 1), (1, 2), (0, 3), (3, 4), (4, 5), (5, 6)
        ], dtype=torch.long).t()


        # Group by `image_id` and create sequences
        self.grouped_data = self.annotations.groupby('image_id', sort=False)
        self.sequences = []
        for _, group in self.grouped_data:
            group = group.reset_index(drop=True)  # Ensure indices align
            for i in range(len(group) - self.time_steps + 1):
                self.sequences.append(group.iloc[i:i + self.time_steps])

    def __len__(self) -> int:
        """
        Return the total number of sequences.
        """
        return len(self.sequences)

    def __getitem__(self, idx: int) -> tuple:
        """
        Fetch a sequence by index.

        Args:
            idx (int): Index of the sequence.

        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]: 
                - Image sequence tensor: [T, C, H, W]
                - Keypoints tensor: [T, K, 2]
                - Edges tensor: [2, num_edges]
                - Behavior label tensor: []
        """
        # Extract the sequence
        sequence = self.sequences[idx]

        # Load behavior label
        behavior_label = sequence.iloc[0]['behavior']
        label = torch.tensor(behavior_label, dtype=torch.long)

        # Load images and keypoints for the sequence
        images = []
        keypoints = []

        for _, row in sequence.iterrows():
            # Image path
            img_path = os.path.join(self.dataset_root_folder, self.label_mapping[row['behavior']], row['image_id'],row['image_file'])
            image = Image.open(img_path).convert("RGB")

            # Apply image transform if provided
            if self.img_transform:
                image = self.img_transform(image)
            images.append(image)

            # Extract keypoints and reshape as [K, 2]
            keypoint_columns = [
                'head_x', 'head_y',
                'beak_base_x', 'beak_base_y',
                'beak_tip_x', 'beak_tip_y',
                'neck_x', 'neck_y',
                'body1_x', 'body1_y',
                'body2_x', 'body2_y',
                'tail_base_x', 'tail_base_y'
            ]

            keypoint_values = row[keypoint_columns].values.astype('float32')
            keypoint_tensor = torch.tensor(keypoint_values, dtype=torch.float32).view(-1, 2)

            # Apply keypoint transform if provided
            if self.kp_transform:
                keypoint_tensor = self.kp_transform(keypoint_tensor)
            keypoints.append(keypoint_tensor)

        # Stack images and keypoints
        images = torch.stack(images)  # Shape: [T, C, H, W]
        keypoints = torch.stack(keypoints)  # Shape: [T, K, 2]

        return images, keypoints, self.edges, label



class BirdBehaviorClassifier(nn.Module):
    def __init__(self, num_keypoints: int, hidden_dim: int = 128, name = "gcn_gru_relu_dropout"):
        """
        Classifier model for bird behavior using GCN and GRU modules.

        Args:
            num_keypoints (int): Number of keypoints in input.
            hidden_dim (int, optional): Dimension of hidden layers. Default is 128.
        """
        super(BirdBehaviorClassifier, self).__init__()
        self.name = name
        self.num_keypoints = num_keypoints
        self.hidden_dim = hidden_dim

        # Spatial GCN for keypoints
        self.gcn = gnn.GCNConv(num_keypoints * 2, hidden_dim)

        # Temporal GRU for sequential data
        self.temporal_gru = nn.GRU(hidden_dim, hidden_dim, batch_first=True)
        self.dropout = nn.Dropout(0.3)

        # Classification head
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1)  # Binary classification (logits for BCEWithLogitsLoss)
        )

    def forward(self, keypoints: torch.Tensor, edge_index: torch.Tensor):
        """
        Forward pass of the Bird Behavior Classifier.

        Args:
            keypoints (torch.Tensor): Tensor of shape [B, T, K, 2] where:
                                      B = batch size,
                                      T = time steps,
                                      K = keypoints,
                                      2 = (x, y) coordinates.
            edge_index (torch.Tensor): Edge index tensor for GCN, shape [2, num_edges].

        Returns:
            torch.Tensor: Logits for binary classification, shape [B, 1].
        """
        # Flatten batch and time for GCN input
        batch_size, time_steps, num_keypoints, _ = keypoints.shape
        keypoints = keypoints.view(batch_size * time_steps, num_keypoints * 2)  # [B*T, K*2]

        # Spatial GCN
        gcn_out = self.gcn(keypoints, edge_index)  # [B*T, hidden_dim]

        # Reshape for temporal GRU
        gcn_out = gcn_out.view(batch_size, time_steps, self.hidden_dim)  # [B, T, hidden_dim]

        # Temporal GRU
        temporal_out, _ = self.temporal_gru(gcn_out)  # [B, T, hidden_dim]

        # Global Average Pooling across time
        pooled_out = temporal_out.mean(dim=1)  # [B, hidden_dim]

        # Classification head
        logits = self.fc(pooled_out).float()  # [B, 1]
        return logits



def evaluate(model, dataloader, device, criterion, edge_index, debug=False):
    """
    Evaluate the BirdBehaviorClassifier on a validation/test dataset.

    Args:
        model (nn.Module): The BirdBehaviorClassifier model.
        dataloader (DataLoader): DataLoader for the validation or test dataset.
        device (torch.device): Device to perform the computation on ('cuda' or 'cpu').
        criterion (nn.Module): Loss function to calculate loss during evaluation.
        edge_index (torch.Tensor): Edge index tensor for the graph structure in the model.
        debug (bool): If True, prints debug information about predictions and labels.

    Returns:
        dict: Evaluation metrics including accuracy, loss, precision, recall, and F1-score.
    """
    model.eval()  # Set the model to evaluation mode
    total_samples = 0
    correct_predictions = 0
    total_loss = 0.0

    all_targets = []
    all_predictions = []

    # Use torch.inference_mode() for evaluation to save memory and computations
    with torch.inference_mode():
        sample = 0
        for batch in dataloader:
            # if sample == 7:  # Limit to 7 batches for debugging purposes
            #     break

            images, keypoints, _, labels = batch  # Only use keypoints and labels
            keypoints, labels = keypoints.to(device), labels.to(device)

            # Ensure edge_index is on the same device as the model
            edge_index = edge_index.to(device)

            # Forward pass
            outputs = model(keypoints, edge_index).squeeze(1)  # Outputs logits

            # Compute loss
            loss = criterion(outputs, labels.float())  # Ensure correct data type
            total_loss += loss.item()

            # Compute predictions (logits > 0.0 maps to class 1)
            predicted_classes = (outputs > 0).int()  # Threshold at 0

            # Debugging logs
            if debug:
                print(f"Expected: {labels}")
                print(f"Prediction: {predicted_classes}")
                print('-' * 100)

            # Update metrics
            total_samples += labels.size(0)
            correct_predictions += (predicted_classes == labels).sum().item()

            # Store for precision/recall calculation
            all_targets.extend(labels.cpu().numpy())
            all_predictions.extend(predicted_classes.cpu().numpy())

            sample += 1

    accuracy = correct_predictions / total_samples if total_samples > 0 else 0.0

    precision, recall, f1_score, _ = precision_recall_fscore_support(
        all_targets, all_predictions, average='binary', zero_division=0
    )

    # Return evaluation metrics
    return {
        'accuracy': accuracy,
        'loss': total_loss / len(dataloader) if len(dataloader) > 0 else 0.0,
        'precision': precision,
        'recall': recall,
        'f1_score': f1_score
    }