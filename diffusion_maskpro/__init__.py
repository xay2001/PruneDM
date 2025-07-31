"""
Diffusion MaskPro: N:M Sparsity Learning for Diffusion Models

This module implements MaskPro's probabilistic N:M sparsity learning
adapted for diffusion models, supporting both Conv2d and Linear layers.
"""

from .maskpro_layer import MaskProLayer
from .conv_adapter import (
    reshape_conv_for_nm_sparsity,
    apply_nm_mask_to_conv,
    validate_nm_compatibility,
    suggest_nm_pattern,
    analyze_layer_nm_efficiency
)
from .utils import (
    wrap_model_with_maskpro,
    extract_mask_parameters,
    count_nm_sparsity,
    get_model_sparsity_summary,
    validate_maskpro_model,
    load_maskpro_state
)
from .maskpro_trainer import DiffusionMaskProTrainer

__version__ = "0.1.0"
__author__ = "Diffusion-MaskPro Integration Team"

__all__ = [
    'MaskProLayer',
    'reshape_conv_for_nm_sparsity',
    'apply_nm_mask_to_conv',
    'validate_nm_compatibility',
    'suggest_nm_pattern',
    'analyze_layer_nm_efficiency',
    'wrap_model_with_maskpro',
    'extract_mask_parameters',
    'count_nm_sparsity',
    'get_model_sparsity_summary',
    'validate_maskpro_model',
    'load_maskpro_state',
    'DiffusionMaskProTrainer'
] 