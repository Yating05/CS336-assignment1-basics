"""
Data loading utilities for CS336 Assignment 1.
"""

import numpy as np
import torch
from typing import Tuple


def get_batch(
    x: np.ndarray,
    batch_size: int,
    context_length: int,
    device: str
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Sample a batch of input sequences and corresponding next-token targets.
    
    Args:
        x: Integer array with token IDs
        batch_size: Number of sequences to sample
        context_length: Length of each sequence
        device: PyTorch device string (e.g., 'cpu' or 'cuda:0')
        
    Returns:
        Tuple of (input_sequences, targets) where:
        - input_sequences: Tensor of shape (batch_size, context_length) with input token IDs
        - targets: Tensor of shape (batch_size, context_length) with next-token target IDs
    """
    # We need context_length + 1 tokens to create input and target sequences
    # input: tokens[i:i+context_length]
    # target: tokens[i+1:i+context_length+1]
    
    # Calculate the maximum starting index for sampling
    max_start_idx = len(x) - context_length - 1
    
    if max_start_idx < 0:
        raise ValueError(f"Dataset too small: need at least {context_length + 1} tokens, got {len(x)}")
    
    # Randomly sample starting indices for each sequence in the batch
    start_indices = np.random.randint(0, max_start_idx + 1, size=batch_size)
    
    # Create input and target sequences
    input_sequences = []
    targets = []
    
    for start_idx in start_indices:
        # Input sequence: tokens from start_idx to start_idx + context_length
        input_seq = x[start_idx:start_idx + context_length]
        
        # Target sequence: tokens from start_idx + 1 to start_idx + context_length + 1
        target_seq = x[start_idx + 1:start_idx + context_length + 1]
        
        input_sequences.append(input_seq)
        targets.append(target_seq)
    
    # Convert to numpy arrays and then to PyTorch tensors
    input_sequences = np.array(input_sequences)
    targets = np.array(targets)
    
    # Convert to PyTorch tensors and move to the specified device
    input_tensor = torch.from_numpy(input_sequences).long().to(device)
    target_tensor = torch.from_numpy(targets).long().to(device)
    
    return input_tensor, target_tensor
