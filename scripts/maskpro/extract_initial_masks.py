#!/usr/bin/env python3
"""
Extract Initial Masks from Pruned Diffusion Models

This script extracts initial N:M sparse masks from magnitude-pruned diffusion models
to serve as starting points for MaskPro learning. It handles both DDPM and LDM models.
"""

import torch
import torch.nn as nn
import os
import sys
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import json

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from diffusers import DDPMPipeline, DDIMPipeline, UNet2DModel
from diffusion_maskpro import (
    validate_nm_compatibility, 
    suggest_nm_pattern,
    analyze_layer_nm_efficiency
)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Extract initial masks from pruned diffusion models")
    
    # Input/Output paths
    parser.add_argument("--pruned_model_path", type=str, required=True,
                      help="Path to the magnitude-pruned model directory")
    parser.add_argument("--output_dir", type=str, default="run/maskpro/initial_masks",
                      help="Directory to save extracted masks")
    
    # N:M sparsity configuration
    parser.add_argument("--n", type=int, default=2,
                      help="Number of non-zero weights per group")
    parser.add_argument("--m", type=int, default=4,
                      help="Group size for N:M sparsity")
    
    # Layer selection
    parser.add_argument("--target_layers", nargs="+", 
                      default=["conv", "linear", "to_q", "to_k", "to_v", "proj"],
                      help="Layer name patterns to target for mask extraction")
    parser.add_argument("--exclude_layers", nargs="+",
                      default=["norm", "embed", "pos_embed", "time_embed", "conv_out"],
                      help="Layer name patterns to exclude")
    
    # Processing options
    parser.add_argument("--device", type=str, default="cuda:0",
                      help="Device to use for processing")
    parser.add_argument("--force_extract", action="store_true",
                      help="Force extraction even for incompatible layers")
    parser.add_argument("--save_analysis", action="store_true",
                      help="Save detailed analysis of extracted masks")
    
    return parser.parse_args()


class DiffusionMaskExtractor:
    """
    Extracts initial N:M sparse masks from pruned diffusion models.
    
    This class handles the conversion from magnitude-pruned (structured) models
    to N:M sparse masks that can be used as initialization for MaskPro learning.
    """
    
    def __init__(self, n: int = 2, m: int = 4, device: str = "cuda:0"):
        self.n = n
        self.m = m
        self.device = torch.device(device)
        self.extraction_log = {}
        self.analysis_results = {}
        
    def load_pruned_model(self, model_path: str) -> nn.Module:
        """
        Load a magnitude-pruned diffusion model.
        
        Args:
            model_path: Path to the pruned model directory
            
        Returns:
            The loaded UNet model
        """
        print(f"Loading pruned model from: {model_path}")
        
        # For magnitude-pruned models, directly load from pruned directory
        # since the structure has been changed and won't match original pipeline config
        pruned_dir = os.path.join(model_path, "pruned")
        
        if os.path.exists(pruned_dir):
            # Try different naming conventions
            potential_files = [
                "unet_pruned.pth",
                "unet_ema_pruned.pth", 
                "model_pruned.pth"
            ]
            
            model_file = None
            for file_name in potential_files:
                file_path = os.path.join(pruned_dir, file_name)
                if os.path.exists(file_path):
                    model_file = file_path
                    break
            
            if model_file is None:
                raise FileNotFoundError(f"No pruned model file found in {pruned_dir}")
            
            print(f"Loading direct pruned model from: {model_file}")
            model = torch.load(model_file, map_location='cpu')
            print("✓ Loaded direct pruned model")
            
        else:
            # Fallback: try to load as pipeline with ignore_mismatched_sizes
            print("No pruned directory found, trying pipeline loading with size mismatch handling...")
            try:
                if "ddpm" in model_path.lower():
                    pipeline = DDPMPipeline.from_pretrained(
                        model_path, 
                        low_cpu_mem_usage=False,
                        ignore_mismatched_sizes=True
                    )
                else:
                    pipeline = DDIMPipeline.from_pretrained(
                        model_path,
                        low_cpu_mem_usage=False, 
                        ignore_mismatched_sizes=True
                    )
                model = pipeline.unet
                print("✓ Loaded as pipeline with size mismatch handling")
            except Exception as e:
                raise ValueError(f"Failed to load model: {e}")
        
        model = model.to(self.device)
        model.eval()
        
        # Print model info
        total_params = sum(p.numel() for p in model.parameters())
        print(f"Model loaded: {total_params:,} parameters")
        
        return model
    
    def should_extract_layer(self, name: str, layer: nn.Module, 
                           target_patterns: List[str], 
                           exclude_patterns: List[str]) -> Tuple[bool, str]:
        """
        Determine if a layer should have its mask extracted.
        
        Args:
            name: Layer name
            layer: Layer module
            target_patterns: Patterns to include
            exclude_patterns: Patterns to exclude
            
        Returns:
            Tuple of (should_extract, reason)
        """
        # Check layer type
        if not isinstance(layer, (nn.Conv2d, nn.Linear)):
            return False, f"Layer type {type(layer).__name__} not supported"
        
        # Check exclusion patterns
        for pattern in exclude_patterns:
            if pattern.lower() in name.lower():
                return False, f"Matched exclusion pattern: {pattern}"
        
        # Check inclusion patterns
        included = False
        for pattern in target_patterns:
            if pattern.lower() in name.lower():
                included = True
                break
        
        if not included:
            return False, "No target pattern matched"
        
        # Check N:M compatibility
        is_compatible, reason = validate_nm_compatibility(layer, self.n, self.m)
        if not is_compatible:
            return False, f"N:M incompatible: {reason}"
        
        return True, "Compatible and targeted"
    
    def extract_mask_from_layer(self, layer: nn.Module, layer_name: str) -> Tuple[torch.Tensor, dict]:
        """
        Extract N:M sparse mask from a single layer.
        
        Args:
            layer: The layer to extract mask from
            layer_name: Name of the layer
            
        Returns:
            Tuple of (mask_tensor, extraction_info)
        """
        weight = layer.weight.data
        original_shape = weight.shape
        
        # Reshape for N:M grouping (input-channel-wise)
        if isinstance(layer, nn.Conv2d):
            # (out_ch, in_ch, h, w) -> (out_ch, in_ch*h*w)
            reshaped_weight = weight.view(weight.size(0), -1)
        else:  # Linear
            # Already in correct shape
            reshaped_weight = weight
        
        # Group into M-sized chunks
        num_groups = reshaped_weight.size(1) // self.m
        if reshaped_weight.size(1) % self.m != 0:
            raise ValueError(f"Layer {layer_name}: dimension not divisible by M={self.m}")
        
        # Reshape to (out_features, num_groups, M)
        grouped_weight = reshaped_weight[:, :num_groups * self.m].view(
            reshaped_weight.size(0), num_groups, self.m
        )
        
        # For each group, find top-N weights by magnitude
        abs_weight = torch.abs(grouped_weight)
        _, top_indices = torch.topk(abs_weight, k=self.n, dim=2)
        
        # Create binary mask
        mask_groups = torch.zeros_like(grouped_weight, dtype=torch.bool)
        mask_groups.scatter_(2, top_indices, True)
        
        # Reshape back to original weight shape
        mask_flat = mask_groups.view(reshaped_weight.size(0), -1)
        
        # Handle any remaining dimensions that weren't divisible by M
        if reshaped_weight.size(1) % self.m != 0:
            remaining_dims = reshaped_weight.size(1) % self.m
            remaining_mask = torch.ones(reshaped_weight.size(0), remaining_dims, dtype=torch.bool)
            mask_flat = torch.cat([mask_flat, remaining_mask], dim=1)
        
        # Reshape to original weight shape
        mask = mask_flat.view(original_shape)
        
        # Calculate extraction statistics
        total_weights = mask.numel()
        nonzero_weights = mask.sum().item()
        sparsity_ratio = 1.0 - (nonzero_weights / total_weights)
        
        # Check N:M compliance
        compliance_check = self._check_nm_compliance(mask)
        
        extraction_info = {
            'layer_type': type(layer).__name__,
            'original_shape': original_shape,
            'total_weights': total_weights,
            'nonzero_weights': nonzero_weights,
            'sparsity_ratio': sparsity_ratio,
            'nm_pattern': f"{self.n}:{self.m}",
            'nm_compliance': compliance_check,
            'num_groups': num_groups
        }
        
        return mask, extraction_info
    
    def _check_nm_compliance(self, mask: torch.Tensor) -> float:
        """Check N:M compliance for extracted mask."""
        if isinstance(mask, torch.Tensor) and len(mask.shape) == 4:
            # Conv2d layer
            reshaped = mask.view(mask.size(0), -1)
        else:
            # Linear layer
            reshaped = mask
        
        # Check groups
        num_groups = reshaped.size(1) // self.m
        if num_groups == 0:
            return 0.0
        
        grouped = reshaped[:, :num_groups * self.m].view(reshaped.size(0), num_groups, self.m)
        nonzeros_per_group = grouped.sum(dim=2)
        compliant_groups = (nonzeros_per_group == self.n).float().mean()
        
        return compliant_groups.item()
    
    def extract_masks_from_model(self, model: nn.Module, 
                                target_patterns: List[str],
                                exclude_patterns: List[str],
                                force_extract: bool = False) -> Dict[str, torch.Tensor]:
        """
        Extract masks from all compatible layers in the model.
        
        Args:
            model: The pruned model
            target_patterns: Layer patterns to include
            exclude_patterns: Layer patterns to exclude
            force_extract: Force extraction even for incompatible layers
            
        Returns:
            Dictionary mapping layer names to extracted masks
        """
        extracted_masks = {}
        
        print(f"Extracting {self.n}:{self.m} masks from model layers...")
        print(f"Target patterns: {target_patterns}")
        print(f"Exclude patterns: {exclude_patterns}")
        
        for name, module in model.named_modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                should_extract, reason = self.should_extract_layer(
                    name, module, target_patterns, exclude_patterns
                )
                
                self.extraction_log[name] = reason
                
                if should_extract or force_extract:
                    try:
                        mask, info = self.extract_mask_from_layer(module, name)
                        extracted_masks[name] = mask
                        self.analysis_results[name] = info
                        
                        print(f"✓ {name}: {info['sparsity_ratio']:.1%} sparse, "
                              f"{info['nm_compliance']:.1%} N:M compliant")
                        
                    except Exception as e:
                        error_msg = f"Extraction failed: {str(e)}"
                        self.extraction_log[name] = error_msg
                        print(f"✗ {name}: {error_msg}")
                else:
                    print(f"- {name}: {reason}")
        
        print(f"\nExtracted masks from {len(extracted_masks)} layers")
        return extracted_masks
    
    def save_masks(self, masks: Dict[str, torch.Tensor], output_dir: str):
        """
        Save extracted masks to disk.
        
        Args:
            masks: Dictionary of layer names to masks
            output_dir: Directory to save masks
        """
        os.makedirs(output_dir, exist_ok=True)
        
        print(f"Saving masks to: {output_dir}")
        
        for layer_name, mask in masks.items():
            # Clean layer name for filename
            safe_name = layer_name.replace(".", "_").replace("/", "_")
            mask_file = os.path.join(output_dir, f"{safe_name}.pt")
            
            torch.save(mask, mask_file)
            print(f"  Saved: {safe_name}.pt")
        
        # Save extraction log
        log_file = os.path.join(output_dir, "extraction_log.json")
        with open(log_file, 'w') as f:
            # Convert any tensor values to serializable format
            serializable_log = {}
            for k, v in self.extraction_log.items():
                if isinstance(v, torch.Tensor):
                    serializable_log[k] = v.tolist()
                else:
                    serializable_log[k] = str(v)
            json.dump(serializable_log, f, indent=2)
        
        print(f"  Saved: extraction_log.json")
    
    def save_analysis(self, output_dir: str):
        """Save detailed analysis of extracted masks."""
        if not self.analysis_results:
            print("No analysis results to save")
            return
        
        analysis_file = os.path.join(output_dir, "mask_analysis.json")
        
        # Convert tensor data to serializable format
        serializable_analysis = {}
        for layer_name, info in self.analysis_results.items():
            serializable_info = {}
            for k, v in info.items():
                if isinstance(v, torch.Size):
                    serializable_info[k] = list(v)
                elif isinstance(v, torch.Tensor):
                    serializable_info[k] = v.item() if v.numel() == 1 else v.tolist()
                else:
                    serializable_info[k] = v
            serializable_analysis[layer_name] = serializable_info
        
        with open(analysis_file, 'w') as f:
            json.dump(serializable_analysis, f, indent=2)
        
        print(f"  Saved: mask_analysis.json")
        
        # Print summary
        total_layers = len(self.analysis_results)
        avg_sparsity = sum(info['sparsity_ratio'] for info in self.analysis_results.values()) / total_layers
        avg_compliance = sum(info['nm_compliance'] for info in self.analysis_results.values()) / total_layers
        
        print(f"\nMask Extraction Summary:")
        print(f"  Total layers processed: {total_layers}")
        print(f"  Average sparsity: {avg_sparsity:.1%}")
        print(f"  Average N:M compliance: {avg_compliance:.1%}")


def main():
    """Main extraction process."""
    args = parse_args()
    
    print("=" * 60)
    print("🎭 Diffusion MaskPro: Initial Mask Extraction")
    print("=" * 60)
    
    # Create extractor
    extractor = DiffusionMaskExtractor(n=args.n, m=args.m, device=args.device)
    
    try:
        # Load pruned model
        model = extractor.load_pruned_model(args.pruned_model_path)
        
        # Extract masks
        masks = extractor.extract_masks_from_model(
            model=model,
            target_patterns=args.target_layers,
            exclude_patterns=args.exclude_layers,
            force_extract=args.force_extract
        )
        
        if not masks:
            print("⚠️  No masks extracted. Check compatibility and layer patterns.")
            return
        
        # Save masks
        extractor.save_masks(masks, args.output_dir)
        
        # Save analysis if requested
        if args.save_analysis:
            extractor.save_analysis(args.output_dir)
        
        print("\n✅ Mask extraction completed successfully!")
        print(f"Masks saved to: {args.output_dir}")
        
    except Exception as e:
        print(f"❌ Mask extraction failed: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code) 