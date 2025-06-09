import numpy as np
import torch
import os
import pandas as pd

def make_directory_if_not_exist(path):
    if not os.path.exists(path):
        os.makedirs(path)
    
def remap_labels(y):
    """
    Remap labels in tensor y to be in the range [0, num_classes - 1].
    
    Args:
        y (torch.Tensor): Tensor of original labels (any integer values).
        
    Returns:
        torch.Tensor: Remapped labels in the range [0, num_classes - 1].
        dict: Mapping from original labels to new indices.
    """
    unique_labels = y.unique()  # Find all unique labels
    label_mapping = {label.item(): idx for idx, label in enumerate(unique_labels)}  # Create mapping
    
    # Create a copy of y with mapped labels
    y_mapped = y.clone()
    for original_label, new_label in label_mapping.items():
        y_mapped[y == original_label] = new_label  # Apply mapping
    
    return y_mapped, label_mapping

def sigmoid_rampup(epoch, max_epoch=30, max_weight=1.0):
    """Sigmoid-based ramp-up function"""
    if epoch >= max_epoch:
        return max_weight
    return max_weight * float(np.exp(-5 * (1 - epoch / max_epoch) ** 2))