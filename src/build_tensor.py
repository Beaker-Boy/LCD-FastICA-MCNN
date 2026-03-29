import torch
from torch.utils.data import random_split, TensorDataset
import numpy as np
from scipy.io import loadmat
import os

def build_tensor_data(file_list, label_list, output_dir, sample_length=1024, samples_per_file=100, train_split_ratio=0.8):
    """
    从 .mat 文件列表构建、划分并保存训练和验证数据集。

    Args:
        file_list (list): a list of .mat file paths.
        label_list (list): A corresponding list of labels for each file.
        output_dir (str): Directory to save the output .pt files.
        sample_length (int): The length of each data sample.
        samples_per_file (int): How many samples to extract from each file.
        train_split_ratio (float): The ratio of training data.
    """
    if not file_list:
        raise ValueError("Input file list cannot be empty.")
    if len(file_list) != len(label_list):
        raise ValueError("File list and label list must have the same length.")

    all_samples = []
    all_labels = []

    print("Starting data collection from .mat files...")
    for file_path, label in zip(file_list, label_list):
        try:
            data = loadmat(file_path)
            # shape (length, channels)
            signal = data['ICA_Components']
        except Exception as e:
            print(f"Warning: Could not load or process file {file_path}. Error: {e}")
            continue

        if signal.shape[0] < sample_length:
            print(f"Warning: Signal in {file_path} is shorter ({signal.shape[0]}) than sample_length ({sample_length}). Skipping.")
            continue
            
        # Generate samples by taking random slices
        for _ in range(samples_per_file):
            start_index = np.random.randint(0, signal.shape[0] - sample_length + 1)
            sample = signal[start_index : start_index + sample_length]
            all_samples.append(sample)
            all_labels.append(label)

    if not all_samples:
        raise ValueError("No valid samples could be generated from the provided files.")

    print(f"Generated a total of {len(all_samples)} samples.")

    # Convert to numpy arrays and then to tensors
    # Transpose to get shape (samples, channels, length)
    all_samples_np = np.array(all_samples, dtype=np.float32).transpose(0, 2, 1)
    all_labels_np = np.array(all_labels, dtype=np.int64)

    all_samples_tensor = torch.from_numpy(all_samples_np)
    all_labels_tensor = torch.from_numpy(all_labels_np)

    # Create a full dataset for splitting
    full_dataset = TensorDataset(all_samples_tensor, all_labels_tensor)

    # Split into training and validation sets
    train_size = int(train_split_ratio * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

    # Separate data and labels
    train_data, train_labels = zip(*[(data, label) for data, label in train_dataset])
    val_data, val_labels = zip(*[(data, label) for data, label in val_dataset])

    train_data_tensor = torch.stack(train_data)
    train_labels_tensor = torch.stack(train_labels)
    val_data_tensor = torch.stack(val_data)
    val_labels_tensor = torch.stack(val_labels)

    # --- Proper Normalization ---
    # Calculate mean and std ONLY from the training data
    # Reshape to (channels, data) to calculate statistics per channel
    num_channels = train_data_tensor.shape[1]
    train_data_reshaped = train_data_tensor.permute(1, 0, 2).reshape(num_channels, -1)
    
    mean = train_data_reshaped.mean(dim=1).reshape(1, num_channels, 1)
    std = train_data_reshaped.std(dim=1).reshape(1, num_channels, 1)

    # Apply normalization to both train and validation data
    train_data_tensor = (train_data_tensor - mean) / (std + 1e-8) # Add epsilon to avoid division by zero
    val_data_tensor = (val_data_tensor - mean) / (std + 1e-8)
    
    print("Normalization complete.")
    print(f"Training data shape: {train_data_tensor.shape}")
    print(f"Validation data shape: {val_data_tensor.shape}")

    # Save tensors to output directory
    os.makedirs(output_dir, exist_ok=True)
    train_data_path = os.path.join(output_dir, "train_tensor_data.pt")
    val_data_path = os.path.join(output_dir, "val_tensor_data.pt")
    train_labels_path = os.path.join(output_dir, "train_labels.pt")
    val_labels_path = os.path.join(output_dir, "val_labels.pt")

    torch.save(train_data_tensor, train_data_path)
    torch.save(val_data_tensor, val_data_path)
    torch.save(train_labels_tensor, train_labels_path)
    torch.save(val_labels_tensor, val_labels_path)

    print(f"Successfully saved datasets to {output_dir}")
    
    # Return paths for confirmation
    return [train_data_path, val_data_path, train_labels_path, val_labels_path]
