"""
Utility functions for Diffusion MaskPro

This module provides utility functions for model wrapping, parameter extraction,
and sparsity analysis for the Diffusion MaskPro framework.
"""

import torch
import torch.nn as nn
from typing import List, Dict, Tuple, Optional, Union, Callable
from .maskpro_layer import MaskProLayer
from .conv_adapter import validate_nm_compatibility, suggest_nm_pattern
import re


def wrap_model_with_maskpro(
    model: nn.Module,
    n: int = 2,
    m: int = 4,
    target_layers: Optional[List[str]] = None,
    exclude_layers: Optional[List[str]] = None,
    layer_types: Tuple[type, ...] = (nn.Conv2d, nn.Linear),
    **maskpro_kwargs
) -> Dict[str, str]:
    """
    Wrap specified layers in a model with MaskPro layers for N:M sparsity learning.
    
    Args:
        model: The model to wrap
        n: Number of non-zero weights per group
        m: Group size for N:M sparsity
        target_layers: List of layer name patterns to include (regex supported)
        exclude_layers: List of layer name patterns to exclude (regex supported)
        layer_types: Tuple of layer types to consider for wrapping
        **maskpro_kwargs: Additional arguments for MaskProLayer
        
    Returns:
        Dictionary mapping original layer names to wrapping status
    """
    wrap_log = {}
    
    # Default target patterns for diffusion models
    if target_layers is None:
        target_layers = [
            r".*conv.*",      # All conv layers
            r".*linear.*",    # All linear layers  
            r".*to_[qkv].*",  # Attention projections
            r".*proj_.*",     # Projection layers
        ]
    
    # Default exclusion patterns
    if exclude_layers is None:
        exclude_layers = [
            r".*norm.*",      # Normalization layers
            r".*embed.*",     # Embedding layers
            r".*pos_embed.*", # Positional embeddings
            r".*cls_token.*", # Class tokens
            r".*conv_out.*",  # Output convolutions (often crucial)
        ]
    
    def should_wrap_layer(name: str, layer: nn.Module) -> Tuple[bool, str]:
        """Determine if a layer should be wrapped."""
        # Check layer type
        if not isinstance(layer, layer_types):
            return False, f"Layer type {type(layer)} not in target types"
        
        # Check exclusion patterns
        for exclude_pattern in exclude_layers:
            if re.match(exclude_pattern, name):
                return False, f"Matched exclusion pattern: {exclude_pattern}"
        
        # Check inclusion patterns
        included = False
        for target_pattern in target_layers:
            if re.match(target_pattern, name):
                included = True
                break
        
        if not included:
            return False, "No target pattern matched"
        
        # Check N:M compatibility
        is_compatible, reason = validate_nm_compatibility(layer, n, m)
        if not is_compatible:
            return False, f"N:M incompatible: {reason}"
        
        return True, "Compatible and targeted"
    
    # Recursively wrap layers
    def wrap_recursive(module: nn.Module, prefix: str = ""):
        for name, child in list(module.named_children()):
            full_name = f"{prefix}.{name}" if prefix else name
            
            # Recursively process children
            wrap_recursive(child, full_name)
            
            # Check if this layer should be wrapped
            should_wrap, reason = should_wrap_layer(full_name, child)
            wrap_log[full_name] = reason
            
            if should_wrap:
                try:
                    # Create MaskPro wrapper
                    wrapped_layer = MaskProLayer(child, n=n, m=m, **maskpro_kwargs)
                    
                    # Replace the original layer
                    setattr(module, name, wrapped_layer)
                    
                    wrap_log[full_name] = f"✓ Wrapped successfully with {n}:{m} sparsity"
                    
                except Exception as e:
                    wrap_log[full_name] = f"✗ Wrapping failed: {str(e)}"
    
    # Start wrapping
    wrap_recursive(model)
    
    return wrap_log


def extract_mask_parameters(model: nn.Module) -> Dict[str, torch.Tensor]:
    """
    Extract all mask parameters from a model with MaskPro layers.
    
    Args:
        model: Model containing MaskPro layers
        
    Returns:
        Dictionary mapping parameter names to mask_logits tensors
    """
    mask_params = {}
    
    for name, param in model.named_parameters():
        if 'mask_logits' in name:
            mask_params[name] = param
    
    return mask_params


def get_model_sparsity_summary(model: nn.Module) -> Dict[str, Union[dict, float, int]]:
    """
    Get comprehensive sparsity statistics for a model with MaskPro layers.
    
    Args:
        model: Model to analyze
        
    Returns:
        Dictionary with detailed sparsity information
    """
    summary = {
        'layer_details': {},
        'overall_stats': {
            'total_maskpro_layers': 0,
            'total_params': 0,
            'total_sparse_params': 0,
            'total_non_zero_params': 0,
            'overall_sparsity': 0.0,
            'nm_patterns': {}
        }
    }
    
    maskpro_layers = []
    
    # Collect all MaskPro layers
    for name, module in model.named_modules():
        if isinstance(module, MaskProLayer):
            maskpro_layers.append((name, module))
    
    summary['overall_stats']['total_maskpro_layers'] = len(maskpro_layers)
    
    # Analyze each layer
    for name, layer in maskpro_layers:
        layer_info = layer.get_sparsity_info()
        summary['layer_details'][name] = layer_info
        
        # Update overall stats
        summary['overall_stats']['total_sparse_params'] += layer_info['total_params']
        summary['overall_stats']['total_non_zero_params'] += layer_info['non_zero_params']
        
        # Track N:M patterns
        pattern = layer_info['target_nm_pattern']
        if pattern not in summary['overall_stats']['nm_patterns']:
            summary['overall_stats']['nm_patterns'][pattern] = 0
        summary['overall_stats']['nm_patterns'][pattern] += 1
    
    # Count total model parameters
    total_model_params = sum(p.numel() for p in model.parameters())
    summary['overall_stats']['total_params'] = total_model_params
    
    # Calculate overall sparsity (considering only MaskPro layers)
    if summary['overall_stats']['total_sparse_params'] > 0:
        sparse_ratio = (
            1.0 - summary['overall_stats']['total_non_zero_params'] / 
            summary['overall_stats']['total_sparse_params']
        )
        summary['overall_stats']['overall_sparsity'] = sparse_ratio
    
    return summary


def count_nm_sparsity(model: nn.Module) -> Tuple[int, int, float]:
    """
    Count N:M sparsity statistics across all MaskPro layers.
    
    Args:
        model: Model to analyze
        
    Returns:
        Tuple of (total_params, non_zero_params, sparsity_ratio)
    """
    total_params = 0
    non_zero_params = 0
    
    for module in model.modules():
        if isinstance(module, MaskProLayer):
            info = module.get_sparsity_info()
            total_params += info['total_params']
            non_zero_params += info['non_zero_params']
    
    sparsity_ratio = 1.0 - (non_zero_params / max(total_params, 1))
    return total_params, non_zero_params, sparsity_ratio


def analyze_model_nm_efficiency(model: nn.Module) -> Dict[str, dict]:
    """
    Analyze theoretical efficiency gains for all MaskPro layers in a model.
    
    Args:
        model: Model to analyze
        
    Returns:
        Dictionary mapping layer names to efficiency analysis
    """
    from .conv_adapter import analyze_layer_nm_efficiency
    
    efficiency_analysis = {}
    
    for name, module in model.named_modules():
        if isinstance(module, MaskProLayer):
            # Get the original layer for analysis
            original_layer = module.original_layer
            n, m = module.n, module.m
            
            analysis = analyze_layer_nm_efficiency(original_layer, n, m)
            efficiency_analysis[name] = analysis
    
    return efficiency_analysis


def suggest_nm_patterns_for_model(model: nn.Module) -> Dict[str, List[Tuple[int, int]]]:
    """
    Suggest optimal N:M patterns for all layers in a model.
    
    Args:
        model: Model to analyze
        
    Returns:
        Dictionary mapping layer names to suggested N:M patterns
    """
    suggestions = {}
    
    for name, module in model.named_modules():
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            patterns = suggest_nm_pattern(module)
            if patterns:
                suggestions[name] = patterns
    
    return suggestions


def validate_maskpro_model(model: nn.Module) -> Dict[str, Union[bool, List[str]]]:
    """
    Validate a model with MaskPro layers for correctness.
    
    Args:
        model: Model to validate
        
    Returns:
        Dictionary with validation results
    """
    results = {
        'is_valid': True,
        'errors': [],
        'warnings': [],
        'layer_count': 0,
        'mask_param_count': 0
    }
    
    maskpro_layers = []
    mask_params = 0
    
    # Check MaskPro layers
    for name, module in model.named_modules():
        if isinstance(module, MaskProLayer):
            maskpro_layers.append((name, module))
            
            # Validate layer configuration
            try:
                # Test forward pass with dummy input
                if module.layer_type == 'conv2d':
                    dummy_input = torch.randn(1, module.original_layer.in_channels, 8, 8)
                else:
                    dummy_input = torch.randn(1, module.original_layer.in_features)
                
                with torch.no_grad():
                    _ = module(dummy_input)
                    
            except Exception as e:
                results['errors'].append(f"Layer {name}: Forward pass failed - {str(e)}")
                results['is_valid'] = False
            
            # Check mask shape consistency
            try:
                mask = module.generate_nm_mask(training=False)
                if mask.shape != module.original_weight_shape:
                    results['errors'].append(
                        f"Layer {name}: Mask shape mismatch - "
                        f"expected {module.original_weight_shape}, got {mask.shape}"
                    )
                    results['is_valid'] = False
            except Exception as e:
                results['errors'].append(f"Layer {name}: Mask generation failed - {str(e)}")
                results['is_valid'] = False
    
    # Count mask parameters
    for name, param in model.named_parameters():
        if 'mask_logits' in name:
            mask_params += param.numel()
    
    results['layer_count'] = len(maskpro_layers)
    results['mask_param_count'] = mask_params
    
    # Warnings
    if len(maskpro_layers) == 0:
        results['warnings'].append("No MaskPro layers found in model")
    
    if mask_params == 0:
        results['warnings'].append("No mask parameters found in model")
    
    return results


def save_maskpro_state(
    model: nn.Module, 
    filepath: str,
    include_model_weights: bool = True
) -> None:
    """
    Save MaskPro-specific state information.
    
    Args:
        model: Model with MaskPro layers
        filepath: Path to save state
        include_model_weights: Whether to include full model weights
    """
    state = {
        'mask_parameters': extract_mask_parameters(model),
        'sparsity_summary': get_model_sparsity_summary(model),
        'layer_configurations': {}
    }
    
    # Save layer configurations
    for name, module in model.named_modules():
        if isinstance(module, MaskProLayer):
            state['layer_configurations'][name] = {
                'n': module.n,
                'm': module.m,
                'layer_type': module.layer_type,
                'original_shape': module.original_weight_shape,
                'use_gumbel': module.use_gumbel,
                'temperature': module.temperature
            }
    
    if include_model_weights:
        state['model_state_dict'] = model.state_dict()
    
    torch.save(state, filepath)


def load_maskpro_state(model: nn.Module, filepath: str) -> Dict[str, str]:
    """
    Load MaskPro state into a model.
    
    Args:
        model: Model with MaskPro layers
        filepath: Path to saved state
        
    Returns:
        Dictionary with loading status for each layer
    """
    state = torch.load(filepath, map_location='cpu')
    load_log = {}
    
    # Load mask parameters
    if 'mask_parameters' in state:
        current_mask_params = extract_mask_parameters(model)
        
        for param_name, saved_param in state['mask_parameters'].items():
            if param_name in current_mask_params:
                try:
                    current_mask_params[param_name].data.copy_(saved_param.data)
                    load_log[param_name] = "✓ Loaded successfully"
                except Exception as e:
                    load_log[param_name] = f"✗ Failed to load: {str(e)}"
            else:
                load_log[param_name] = "✗ Parameter not found in current model"
    
    # Load full model state if available
    if 'model_state_dict' in state:
        try:
            model.load_state_dict(state['model_state_dict'], strict=False)
            load_log['full_model'] = "✓ Model weights loaded"
        except Exception as e:
            load_log['full_model'] = f"✗ Model loading failed: {str(e)}"
    
    return load_log 