import os
import numpy as np
import pandas as pd
from PIL import Image

import torch
from torch.utils.data import Dataset

class PoseDataset(Dataset):
    def __init__(self, dataframe, dataset_root_folder, img_transform=None, kp_transform=None):
        self.annotations = dataframe  # Load the pandas DataFrame directly
        self.dataset_root_folder = dataset_root_folder  # Root folder for the dataset
        self.img_transform = img_transform
        self.kp_transform = kp_transform

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        # Construct the image path from behavior, image_id, and image_file columns
        behavior = self.annotations.iloc[idx]['behavior']
        image_id = self.annotations.iloc[idx]['image_id']
        image_file = self.annotations.iloc[idx]['image_file']
        
        # Create the full image path
        img_path = os.path.join(self.dataset_root_folder, behavior, image_id, image_file)
        
        # Load and process the image
        image = Image.open(img_path).convert("RGB")
        
        # Extract the keypoints (head_x, head_y, ..., body2_x, body2_y) as numpy array
        keypoints = self.annotations.iloc[idx, 3:].values.astype('float32')
        
        if self.img_transform:
            image = self.img_transform(image)

        if self.kp_transform:
            keypoints = self.kp_transform(keypoints)
        

        return image, keypoints


def calculate_head_size(keypoints):
    """
    Calculate head size for a batch of flattened keypoints.

    Args:
        keypoints (torch.Tensor): A tensor of shape (batch_size, num_keypoints * 2),
                                  where each row contains flattened 2D coordinates.
                                  Keypoints are arranged as:
    Returns:
        torch.Tensor: A tensor of shape (batch_size,) containing the head size for each sample.
    """
    # Extract batch size and number of keypoints
    batch_size = keypoints.size(0)  # First dimension is the batch size
    num_keypoints = keypoints.size(1) // 2  # Number of keypoints

    # Reshape to (batch_size, num_keypoints, 2)
    keypoints = keypoints.view(batch_size, num_keypoints, 2)

    # Extract head and beak_tip keypoints
    head = keypoints[:, 0, :]       # Shape: (batch_size, 2)
    beak_tip = keypoints[:, 2, :]   # Shape: (batch_size, 2)

    # Calculate Euclidean distance between head and beak_tip
    head_size = torch.norm(head - beak_tip, p=2, dim=1)  # Shape: (batch_size,)

    return head_size


# TODO(Adam-Al-Rahman): Remove the comments after optimizing the training and testing workflow
# When the threshold is 0.2 in the PCKh (Percentage of Correct Keypoints with Head Normalization) calculation,
# it means that a predicted keypoint is considered correct
# if the Euclidean distance between the predicted and ground truth keypoints is less than 20% of the head size.

def pckh(predictions, ground_truth, head_size, threshold=0.2):
    """
    Calculate PCKh (Percentage of Correct Keypoints with Head Normalization) for a batch of predictions.

    Args:
        predictions (Tensor): Predicted keypoints, shape (batch_size, num_keypoints * 2)
        ground_truth (Tensor): Ground truth keypoints, shape (batch_size, num_keypoints * 2)
        head_size (Tensor): Normalizing head size for each sample, shape (batch_size,)
        threshold (float): Normalized distance threshold (percentage of head size)

    Returns:
        float: PCKh metric as a percentage of correct keypoints
    """
    batch_size, num_flattened = predictions.size()
    num_keypoints = num_flattened // 2  # Derive number of keypoints
    
    # Reshape flattened predictions and ground truth to (batch_size, num_keypoints, 2)
    predictions = predictions.view(batch_size, num_keypoints, 2)
    ground_truth = ground_truth.view(batch_size, num_keypoints, 2)
    
    # Calculate Euclidean distance between predicted and ground truth keypoints
    distance = torch.norm(predictions - ground_truth, p=2, dim=2)  # shape: (batch_size, num_keypoints)
    
    # Normalize by head size for PCKh
    normalized_distance = distance / head_size.unsqueeze(1)  # shape: (batch_size, num_keypoints)
    
    # Calculate PCKh: Count keypoints that are within the threshold
    correct_keypoints = (normalized_distance < threshold).float()  # shape: (batch_size, num_keypoints)
    
    # Compute the percentage of correct keypoints
    pckh = correct_keypoints.sum() / (batch_size * num_keypoints) * 100
    
    return pckh.item()


def pe_accuracy(model, dataloader, device):
    """
    Calculate PCKh accuracy for the entire dataset.

    Args:
        model (nn.Module): The pose estimation model
        dataloader (DataLoader): DataLoader providing the dataset
        device (str): Device to run the model on (either 'cuda' or 'cpu')

    Returns:
        float: Average PCKh for the dataset
    """
    model.eval()  # Set model to evaluation mode
    total_pckh = 0.0
    total_samples = 0

    with torch.inference_mode():  # Disable gradient calculation for evaluation
        for images, keypoints in dataloader:
            images = images.to(device)
            keypoints = keypoints.to(device)
            head_sizes = calculate_head_size(keypoints).to(device)
            
            # Predict keypoints
            outputs = model(images)
            
            # Calculate PCKh for the current batch
            batch_pckh = pckh(outputs, keypoints, head_sizes)
            
            total_pckh += batch_pckh * images.size(0)
            total_samples += images.size(0)
            break # SINGLE BATCH

    # Calculate the average PCKh for the dataset
    average_pckh = total_pckh / total_samples
    return average_pckh



# The Average PCKh (Percentage of Correct Keypoints with Head Normalization) being 94% means that on average,
# only 94% of the predicted keypoints are within the specified "threshold=0.2" (e.g., 20% of the head size) across the dataset.

def pe_accuracy(model, dataloader, device):
    """
    Calculate PCKh accuracy for the entire dataset.

    Args:
        model (nn.Module): The pose estimation model
        dataloader (DataLoader): DataLoader providing the dataset
        device (str): Device to run the model on (either 'cuda' or 'cpu')

    Returns:
        float: Average PCKh for the dataset
    """
    model.eval()  # Set model to evaluation mode
    total_pckh = 0.0
    total_samples = 0

    with torch.inference_mode():  # Disable gradient calculation for evaluation
        for images, keypoints in dataloader:
            images = images.to(device)
            keypoints = keypoints.to(device)
            head_sizes = calculate_head_size(keypoints).to(device)
            
            # Predict keypoints
            outputs = model(images)
            
            # Calculate PCKh for the current batch
            batch_pckh = pckh(outputs, keypoints, head_sizes)
            
            total_pckh += batch_pckh * images.size(0)
            total_samples += images.size(0)

    # Calculate the average PCKh for the dataset
    average_pckh = total_pckh / total_samples
    return average_pckh