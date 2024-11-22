import os
import torch
import json
from datetime import datetime, timezone

from .classification import *  # Import everything from classification.py
from .pose_estimation import *  # Import everything from pose_estimation.py


def update_config(config_path, holder, values):

    try:
        with open(config_path, 'r') as file:
            config_data = json.load(file)
    except FileNotFoundError:
        print(f"Error: The file {config_path} was not found.")
        return
    except json.JSONDecodeError:
        print(f"Error: The file {config_path} is not a valid JSON.")
        return

    # Check if `holder` key exists, if not, create it
    if holder not in config_data:
        config_data[holder] = {}

    # Append new values to the `holder` section
    config_data[holder].update(values)

    try:
        with open(config_path, 'w') as file:
            json.dump(config_data, file, indent=4)
        print(f"Updated {config_path} {holder} successfully.")
    except IOError:
        print(f"Error: Unable to write to {config_path}.")


def save_model(model, architecture, model_path, epochs, learning_rate,  model_type="pe"):
    """
    Saves the model's state_dict to a specified directory with a timestamped filename.

    Args:
        model (torch.nn.Module): The PyTorch model to save.
        architecture (str): The architecture name of the model (e.g., 'ResNet50').
        model_path (str): The directory where the model should be saved.
        model_type (str): sp: snow_petrel, pe: pose_estimation
    """
    # Ensure the directory exists
    os.makedirs(model_path, exist_ok=True)

    current_time_local = datetime.now().strftime("%Y-%m-%dT%H-%M")

    # Combine the directory and file name
    file_name = f"sp_{model_type}_{architecture}_epochs{epochs}_lr{learning_rate}_{current_time_local}.pth"
    full_path = os.path.join(model_path, file_name)

    # Save the model state_dict
    torch.save(model.state_dict(), full_path)

    return full_path



