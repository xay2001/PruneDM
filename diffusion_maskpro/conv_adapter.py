"""
Convolutional Layer N:M Sparsity Adapter

This module provides utility functions for adapting convolutional layers
to N:M sparsity patterns, including shape validation and tensor reshaping.
"""

import torch
import torch.nn as nn
from typing import Tuple, List, Optional, Union
import warnings


def validate_nm_compatibility(
    layer: Union[nn.Conv2d, nn.Linear], 
    n: int, 
    m: int,
    strategy: str = "input_channel_wise"
) -> Tuple[bool, str]:
    """
    Validate if a layer is compatible with N:M sparsity.
    
    Args:
        layer: The layer to validate
        n: Number of non-zero weights per group
        m: Group size
        strategy: Grouping strategy ('input_channel_wise' or 'kernel_wise')
        
    Returns:
        Tuple of (is_compatible, reason)
    """
    if not isinstance(layer, (nn.Conv2d, nn.Linear)):
        return False, f"Unsupported layer type: {type(layer)}"
    
    if not (0 < n < m):
        return False, f"Invalid N:M pattern: n={n}, m={m}"
    
    weight_shape = layer.weight.shape
    
    if isinstance(layer, nn.Conv2d):
        out_ch, in_ch, h, w = weight_shape
        
        if strategy == "input_channel_wise":
            # Check if (in_ch * h * w) is divisible by m
            feature_dim = in_ch * h * w
            if feature_dim % m != 0:
                return False, (
                    f"Input channel dimension {feature_dim} = {in_ch}*{h}*{w} "
                    f"not divisible by M={m}"
                )
        elif strategy == "kernel_wise":
            # Check if kernel size is compatible
            kernel_size = h * w
            if kernel_size % m != 0:
                return False, (
                    f"Kernel size {kernel_size} = {h}*{w} not divisible by M={m}"
                )
        else:
            return False, f"Unknown strategy: {strategy}"
            
    else:  # Linear layer
        out_features, in_features = weight_shape
        if in_features % m != 0:
            return False, (
                f"Input features {in_features} not divisible by M={m}"
            )
    
    return True, "Compatible"


def reshape_conv_for_nm_sparsity(
    conv_weight: torch.Tensor, 
    n: int, 
    m: int,
    strategy: str = "input_channel_wise"
) -> torch.Tensor:
    """
    Reshape convolutional weight tensor for N:M sparsity application.
    
    Args:
        conv_weight: Convolutional weight tensor (out_ch, in_ch, h, w)
        n: Number of non-zero weights per group
        m: Group size  
        strategy: Reshaping strategy
        
    Returns:
        Reshaped tensor suitable for N:M grouping
        
    Raises:
        ValueError: If tensor is not compatible with N:M pattern
    """
    if len(conv_weight.shape) != 4:
        raise ValueError(f"Expected 4D tensor, got shape {conv_weight.shape}")
    
    out_ch, in_ch, h, w = conv_weight.shape
    
    if strategy == "input_channel_wise":
        # Reshape to (out_ch, in_ch*h*w) for input-channel grouping
        reshaped = conv_weight.view(out_ch, -1)
        feature_dim = reshaped.size(1)
        
        if feature_dim % m != 0:
            raise ValueError(
                f"Feature dimension {feature_dim} not divisible by M={m}"
            )
            
        return reshaped
        
    elif strategy == "kernel_wise":
        # Reshape to (out_ch*in_ch, h*w) for kernel-wise grouping
        reshaped = conv_weight.view(out_ch * in_ch, h * w)
        kernel_size = reshaped.size(1)
        
        if kernel_size % m != 0:
            raise ValueError(
                f"Kernel size {kernel_size} not divisible by M={m}"
            )
            
        return reshaped
        
    else:
        raise ValueError(f"Unknown strategy: {strategy}")


def apply_nm_mask_to_conv(
    conv_weight: torch.Tensor,
    mask: torch.Tensor,
    original_shape: Tuple[int, ...],
    strategy: str = "input_channel_wise"
) -> torch.Tensor:
    """
    Apply N:M mask to convolutional weight and restore original shape.
    
    Args:
        conv_weight: Reshaped convolutional weight
        mask: N:M sparse mask
        original_shape: Original weight tensor shape
        strategy: Reshaping strategy used
        
    Returns:
        Masked weight tensor with original shape
    """
    # Apply mask
    masked_weight = conv_weight * mask
    
    # Reshape back to original shape
    return masked_weight.view(original_shape)


def analyze_layer_nm_efficiency(
    layer: Union[nn.Conv2d, nn.Linear],
    n: int,
    m: int
) -> dict:
    """
    Analyze the theoretical efficiency gains from N:M sparsity for a layer.
    
    Args:
        layer: The layer to analyze
        n: Number of non-zero weights per group  
        m: Group size
        
    Returns:
        Dictionary with efficiency analysis
    """
    weight_shape = layer.weight.shape
    total_params = layer.weight.numel()
    
    # Calculate theoretical sparsity
    theoretical_sparsity = 1.0 - (n / m)
    remaining_params = int(total_params * (n / m))
    
    # Estimate memory savings (naive)
    memory_reduction = theoretical_sparsity
    
    # Estimate compute savings (depends on hardware support)
    # For NVIDIA sparse tensor cores, compute reduction ≈ sparsity ratio
    compute_reduction = theoretical_sparsity
    
    analysis = {
        'layer_type': type(layer).__name__,
        'weight_shape': weight_shape,
        'total_params': total_params,
        'nm_pattern': f"{n}:{m}",
        'theoretical_sparsity': theoretical_sparsity,
        'remaining_params': remaining_params,
        'params_removed': total_params - remaining_params,
        'memory_reduction_ratio': memory_reduction,
        'compute_reduction_ratio': compute_reduction,
        'hardware_efficiency': _estimate_hardware_efficiency(layer, n, m)
    }
    
    return analysis


def _estimate_hardware_efficiency(
    layer: Union[nn.Conv2d, nn.Linear],
    n: int, 
    m: int
) -> dict:
    """
    Estimate hardware efficiency for different N:M patterns.
    
    Based on NVIDIA Sparse Tensor Core specifications.
    """
    # Common efficient patterns for NVIDIA hardware
    efficient_patterns = {
        (2, 4): "optimal",  # 2:4 is native supported
        (1, 4): "good",     # 1:4 is also well supported  
        (4, 8): "good",     # 4:8 can be efficient
        (1, 8): "moderate", # 1:8 less efficient
        (2, 8): "moderate"  # 2:8 moderate efficiency
    }
    
    pattern = (n, m)
    efficiency_rating = efficient_patterns.get(pattern, "unknown")
    
    # Estimate actual speedup based on pattern and layer type
    if isinstance(layer, nn.Conv2d):
        # Conv2d benefits more from sparse tensor cores
        speedup_multiplier = {
            "optimal": 1.8,
            "good": 1.5,
            "moderate": 1.2,
            "unknown": 1.0
        }
    else:  # Linear
        # Linear layers have good sparse support
        speedup_multiplier = {
            "optimal": 1.9,
            "good": 1.6,
            "moderate": 1.3,
            "unknown": 1.0
        }
    
    theoretical_sparsity = 1.0 - (n / m)
    estimated_speedup = theoretical_sparsity * speedup_multiplier[efficiency_rating]
    
    return {
        'efficiency_rating': efficiency_rating,
        'estimated_speedup': estimated_speedup,
        'hardware_support': pattern in [(2, 4), (1, 4)],
        'recommended': pattern == (2, 4)
    }


def suggest_nm_pattern(layer: Union[nn.Conv2d, nn.Linear]) -> List[Tuple[int, int]]:
    """
    Suggest optimal N:M patterns for a given layer.
    
    Args:
        layer: The layer to analyze
        
    Returns:
        List of recommended (n, m) patterns, ordered by preference
    """
    weight_shape = layer.weight.shape
    
    # Start with hardware-optimal patterns
    candidates = [(2, 4), (1, 4), (4, 8), (1, 8), (2, 8)]
    
    # Filter by compatibility
    compatible_patterns = []
    for n, m in candidates:
        is_compatible, _ = validate_nm_compatibility(layer, n, m)
        if is_compatible:
            compatible_patterns.append((n, m))
    
    # If no standard patterns work, suggest modifications
    if not compatible_patterns:
        warnings.warn(
            f"No standard N:M patterns compatible with layer shape {weight_shape}. "
            "Consider padding or using different layer dimensions."
        )
        
        # Try to find any working pattern
        if isinstance(layer, nn.Conv2d):
            out_ch, in_ch, h, w = weight_shape
            feature_dim = in_ch * h * w
        else:
            feature_dim = weight_shape[1]
            
        # Find divisors of feature dimension
        for m in [4, 8, 16]:
            if feature_dim % m == 0:
                for n in range(1, m):
                    compatible_patterns.append((n, m))
                break
    
    # Sort by preference (hardware efficiency)
    def pattern_score(pattern):
        n, m = pattern
        if pattern == (2, 4):
            return 10  # Best
        elif pattern == (1, 4):
            return 9   # Very good
        elif pattern in [(4, 8), (2, 8)]:
            return 8   # Good
        else:
            return 5   # Moderate
    
    compatible_patterns.sort(key=pattern_score, reverse=True)
    
    return compatible_patterns


def pad_layer_for_nm_compatibility(
    layer: Union[nn.Conv2d, nn.Linear],
    n: int,
    m: int,
    strategy: str = "input_channel_wise"
) -> Union[nn.Conv2d, nn.Linear]:
    """
    Create a padded version of the layer to make it N:M compatible.
    
    Args:
        layer: Original layer
        n: Number of non-zero weights per group
        m: Group size
        strategy: Padding strategy
        
    Returns:
        New layer with padded dimensions
    """
    # This is a more advanced feature that could be implemented
    # for layers that don't naturally fit N:M patterns
    raise NotImplementedError(
        "Layer padding for N:M compatibility not yet implemented. "
        "Consider using compatible layer dimensions instead."
    ) 