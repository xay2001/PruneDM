"""
MaskPro Layer: Unified N:M Sparsity Learning for Conv2d and Linear Layers

This module implements the core MaskProLayer that wraps existing PyTorch layers
to enable learnable N:M sparsity patterns through probabilistic mask learning.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional, Union
import math


class MaskProLayer(nn.Module):
    """
    Universal MaskPro wrapper for both Conv2d and Linear layers.
    
    Implements learnable N:M sparsity through probabilistic mask distributions.
    Key innovation: Handles 4D convolutional tensors by reshaping to 2D for N:M grouping.
    
    Args:
        original_layer: The layer to wrap (nn.Conv2d or nn.Linear)
        n: Number of non-zero weights to keep in each group
        m: Size of each group (e.g., 2:4 sparsity means n=2, m=4)
        logits_init_scale: Initial scaling for mask logits
        use_gumbel: Whether to use Gumbel Softmax for differentiable sampling
        temperature: Temperature for Gumbel Softmax
    """
    
    def __init__(
        self,
        original_layer: Union[nn.Conv2d, nn.Linear],
        n: int = 2,
        m: int = 4,
        logits_init_scale: float = 10.0,
        use_gumbel: bool = True,
        temperature: float = 1.0
    ):
        super(MaskProLayer, self).__init__()
        
        # Validate inputs
        assert isinstance(original_layer, (nn.Conv2d, nn.Linear)), \
            f"Only Conv2d and Linear layers supported, got {type(original_layer)}"
        assert 0 < n < m, f"Invalid N:M pattern: n={n}, m={m}"
        
        # Store configuration
        self.original_layer = original_layer
        self.n = n
        self.m = m
        self.logits_init_scale = logits_init_scale
        self.use_gumbel = use_gumbel
        self.temperature = temperature
        self.layer_type = 'conv2d' if isinstance(original_layer, nn.Conv2d) else 'linear'
        
        # Get weight shape information
        self.original_weight_shape = original_layer.weight.shape
        self.weight_numel = original_layer.weight.numel()
        
        # Calculate reshaped dimensions for N:M grouping
        self.reshaped_dim, self.num_groups = self._calculate_grouping_dims()
        
        # Initialize learnable mask logits
        # Shape: (num_groups, m) - each group has m logits for categorical distribution
        self.mask_logits = nn.Parameter(
            torch.randn(self.num_groups, self.m) * logits_init_scale
        )
        
        # Register buffer for current mask (for evaluation mode)
        self.register_buffer('current_mask', torch.ones_like(original_layer.weight))
        
        # Track sampling history for policy gradient
        self.log_probs_cache = None
        
        # Initialize mask to identity (no sparsity initially)
        self._initialize_mask()
    
    def _calculate_grouping_dims(self) -> Tuple[int, int]:
        """
        Calculate dimensions for reshaping weights into N:M groups.
        
        Strategy: Input-channel-wise grouping (hardware-friendly)
        - Conv2d: (out_ch, in_ch, h, w) -> (out_ch, in_ch*h*w)
        - Linear: (out_features, in_features) -> unchanged
        
        Returns:
            reshaped_dim: Dimension size after reshaping
            num_groups: Number of N:M groups
        """
        if self.layer_type == 'conv2d':
            out_ch, in_ch, h, w = self.original_weight_shape
            reshaped_dim = in_ch * h * w
        else:  # linear
            out_features, in_features = self.original_weight_shape
            reshaped_dim = in_features
            
        # Check if dimension is divisible by M
        if reshaped_dim % self.m != 0:
            raise ValueError(
                f"Layer dimension {reshaped_dim} not divisible by M={self.m}. "
                f"Consider padding or choosing different M value."
            )
        
        num_groups = reshaped_dim // self.m
        total_groups = self.original_weight_shape[0] * num_groups  # out_features/channels * groups_per_feature
        
        return reshaped_dim, total_groups
    
    def _initialize_mask(self):
        """Initialize mask to a valid N:M pattern."""
        with torch.no_grad():
            # Create initial N:M mask based on magnitude
            weight_flat = self._reshape_weight_for_grouping(self.original_layer.weight)
            groups = weight_flat.view(-1, self.m)
            
            # For each group, keep the top-N largest magnitude weights
            _, top_indices = torch.topk(torch.abs(groups), k=self.n, dim=1)
            
            # Create binary mask
            mask_groups = torch.zeros_like(groups, dtype=torch.bool)
            mask_groups.scatter_(1, top_indices, True)
            
            # Reshape back to original weight shape
            mask_flat = mask_groups.view(weight_flat.shape)
            self.current_mask.copy_(self._reshape_mask_to_original(mask_flat))
    
    def _reshape_weight_for_grouping(self, weight: torch.Tensor) -> torch.Tensor:
        """Reshape weight tensor for N:M grouping."""
        if self.layer_type == 'conv2d':
            # (out_ch, in_ch, h, w) -> (out_ch, in_ch*h*w)
            return weight.view(weight.size(0), -1)
        else:
            # Linear layer already in correct shape
            return weight
    
    def _reshape_mask_to_original(self, mask_flat: torch.Tensor) -> torch.Tensor:
        """Reshape flat mask back to original weight shape."""
        return mask_flat.view(self.original_weight_shape)
    
    def generate_nm_mask(self, training: bool = True) -> torch.Tensor:
        """
        Generate N:M sparse mask from learned logits.
        
        Args:
            training: If True, use probabilistic sampling. If False, use deterministic top-k.
            
        Returns:
            Binary mask tensor with same shape as original weight
        """
        if training and self.use_gumbel:
            return self._generate_mask_gumbel()
        else:
            return self._generate_mask_deterministic()
    
    def _generate_mask_gumbel(self) -> torch.Tensor:
        """Generate mask using Gumbel Softmax for differentiable sampling."""
        # Sample from Gumbel Softmax categorical distribution
        gumbel_samples = F.gumbel_softmax(
            self.mask_logits, 
            tau=self.temperature, 
            hard=True, 
            dim=1
        )  # Shape: (num_groups, m)
        
        # Convert to N:M pattern: keep top N from each group
        # Use straight-through estimator for exact N:M compliance
        with torch.no_grad():
            _, top_indices = torch.topk(gumbel_samples, k=self.n, dim=1)
            hard_mask = torch.zeros_like(gumbel_samples)
            hard_mask.scatter_(1, top_indices, 1.0)
        
        # Straight-through: forward hard mask, backward soft mask
        nm_mask = hard_mask + gumbel_samples - gumbel_samples.detach()
        
        # Cache log probabilities for policy gradient
        probs = F.softmax(self.mask_logits, dim=1)
        selected_probs = torch.gather(probs, 1, top_indices)
        self.log_probs_cache = torch.log(selected_probs + 1e-8).sum()
        
        return self._reshape_groups_to_weight_mask(nm_mask)
    
    def _generate_mask_deterministic(self) -> torch.Tensor:
        """Generate mask using deterministic top-k selection."""
        # Simply take top-k logits in each group
        _, top_indices = torch.topk(self.mask_logits, k=self.n, dim=1)
        
        mask_groups = torch.zeros_like(self.mask_logits)
        mask_groups.scatter_(1, top_indices, 1.0)
        
        return self._reshape_groups_to_weight_mask(mask_groups)
    
    def _reshape_groups_to_weight_mask(self, mask_groups: torch.Tensor) -> torch.Tensor:
        """
        Reshape mask from grouped format back to original weight shape.
        
        Args:
            mask_groups: Shape (num_groups, m)
            
        Returns:
            Mask with original weight shape
        """
        if self.layer_type == 'conv2d':
            out_ch = self.original_weight_shape[0]
            groups_per_channel = self.num_groups // out_ch
            
            # Reshape groups back to (out_ch, groups_per_channel, m)
            mask_reshaped = mask_groups.view(out_ch, groups_per_channel, self.m)
            # Flatten groups dimension: (out_ch, groups_per_channel * m)
            mask_flat = mask_reshaped.view(out_ch, -1)
            # Reshape to original conv weight shape
            return mask_flat.view(self.original_weight_shape)
        else:
            # Linear layer: just reshape groups back to weight shape
            return mask_groups.view(self.original_weight_shape)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with dynamic N:M sparsity.
        
        Args:
            x: Input tensor
            
        Returns:
            Output tensor from masked layer computation
        """
        # Generate current mask
        mask = self.generate_nm_mask(training=self.training)
        
        # Apply mask to weights
        masked_weight = self.original_layer.weight * mask
        
        # Store current mask for later inspection
        self.current_mask.copy_(mask.detach())
        
        # Perform the original layer computation with masked weight
        if self.layer_type == 'conv2d':
            return F.conv2d(
                x, 
                masked_weight,
                bias=self.original_layer.bias,
                stride=self.original_layer.stride,
                padding=self.original_layer.padding,
                dilation=self.original_layer.dilation,
                groups=self.original_layer.groups
            )
        else:  # linear
            return F.linear(x, masked_weight, self.original_layer.bias)
    
    def get_mask_loss(self, main_task_loss: torch.Tensor) -> torch.Tensor:
        """
        Compute policy gradient loss for mask learning.
        
        This is the core of MaskPro's learning algorithm.
        Uses REINFORCE with baseline to update mask distribution.
        
        Args:
            main_task_loss: Loss from the main diffusion task
            
        Returns:
            Policy gradient loss for mask parameters
        """
        if self.log_probs_cache is None:
            return torch.tensor(0.0, device=main_task_loss.device)
        
        # Use negative loss as reward (lower loss = higher reward)
        reward = -main_task_loss.detach()
        
        # Compute baseline for variance reduction
        if not hasattr(self, 'baseline'):
            self.baseline = reward.item()
        else:
            # Exponential moving average baseline
            self.baseline = 0.99 * self.baseline + 0.01 * reward.item()
        
        # Policy gradient: REINFORCE with baseline
        advantage = reward - self.baseline
        mask_loss = -(self.log_probs_cache * advantage)
        
        return mask_loss
    
    def get_sparsity_info(self) -> dict:
        """
        Get detailed sparsity information for analysis.
        
        Returns:
            Dictionary with sparsity statistics
        """
        with torch.no_grad():
            total_params = self.current_mask.numel()
            non_zero_params = self.current_mask.sum().item()
            sparsity_ratio = 1.0 - (non_zero_params / total_params)
            
            # Check N:M compliance
            mask_flat = self._reshape_weight_for_grouping(self.current_mask)
            mask_groups = mask_flat.view(-1, self.m)
            actual_n_per_group = mask_groups.sum(dim=1)
            nm_compliance = (actual_n_per_group == self.n).float().mean().item()
            
            return {
                'total_params': total_params,
                'non_zero_params': non_zero_params,
                'sparsity_ratio': sparsity_ratio,
                'target_nm_pattern': f"{self.n}:{self.m}",
                'nm_compliance': nm_compliance,
                'layer_type': self.layer_type,
                'original_shape': self.original_weight_shape
            }
    
    def __repr__(self) -> str:
        return (f"MaskProLayer({self.layer_type}, {self.n}:{self.m} sparsity, "
                f"shape={self.original_weight_shape})") 