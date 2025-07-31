#!/usr/bin/env python3
"""
Diffusion Model Baseline Loss Computation

This script computes baseline losses for diffusion models using initial masks.
These baseline losses are used by MaskPro's policy gradient algorithm to measure
improvement during mask learning.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import sys
import argparse
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import json
from tqdm import tqdm

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from diffusers import DDPMPipeline, DDIMPipeline, DDPMScheduler, DDIMScheduler
from diffusion_maskpro import wrap_model_with_maskpro, load_maskpro_state
import utils


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Compute baseline losses for diffusion models")
    
    # Model and mask paths
    parser.add_argument("--pruned_model_path", type=str, required=True,
                      help="Path to the magnitude-pruned model")
    parser.add_argument("--initial_masks_dir", type=str, required=True,
                      help="Directory containing initial masks")
    
    # Dataset configuration
    parser.add_argument("--dataset", type=str, default="cifar10",
                      choices=["cifar10", "celeba", "custom"],
                      help="Dataset for computing baseline losses")
    parser.add_argument("--dataset_size", type=int, default=512,
                      help="Number of samples to use for baseline computation")
    parser.add_argument("--batch_size", type=int, default=32,
                      help="Batch size for processing")
    
    # Diffusion configuration
    parser.add_argument("--num_train_timesteps", type=int, default=1000,
                      help="Number of training timesteps")
    parser.add_argument("--timestep_sampling", type=str, default="uniform",
                      choices=["uniform", "importance", "early", "late"],
                      help="Strategy for sampling timesteps")
    parser.add_argument("--timestep_subset", type=int, default=None,
                      help="Subset of timesteps to use (None = all)")
    
    # N:M sparsity configuration
    parser.add_argument("--n", type=int, default=2,
                      help="Number of non-zero weights per group")
    parser.add_argument("--m", type=int, default=4,
                      help="Group size for N:M sparsity")
    
    # Processing options
    parser.add_argument("--device", type=str, default="cuda:0",
                      help="Device to use for computation")
    parser.add_argument("--mixed_precision", action="store_true",
                      help="Use mixed precision computation")
    parser.add_argument("--output_file", type=str, default=None,
                      help="Output file for baseline losses")
    
    return parser.parse_args()


class DiffusionBaselineLossComputer:
    """
    Computes baseline losses for diffusion models with initial masks.
    
    This class loads a pruned diffusion model, applies initial masks,
    and computes losses across different timesteps to establish baselines
    for MaskPro's policy gradient learning.
    """
    
    def __init__(self, 
                 pruned_model_path: str,
                 initial_masks_dir: str,
                 n: int = 2,
                 m: int = 4,
                 device: str = "cuda:0",
                 mixed_precision: bool = False):
        self.pruned_model_path = pruned_model_path
        self.initial_masks_dir = initial_masks_dir
        self.n = n
        self.m = m
        self.device = torch.device(device)
        self.mixed_precision = mixed_precision
        
        self.model = None
        self.scheduler = None
        self.pipeline = None
        
    def load_model_and_scheduler(self):
        """Load the pruned model and scheduler."""
        print(f"Loading pruned model from: {self.pruned_model_path}")
        
        # Try to load as pipeline first
        try:
            if "ddpm" in self.pruned_model_path.lower():
                self.pipeline = DDPMPipeline.from_pretrained(self.pruned_model_path)
                self.scheduler = DDPMScheduler.from_pretrained(self.pruned_model_path, subfolder="scheduler")
            else:
                self.pipeline = DDIMPipeline.from_pretrained(self.pruned_model_path)
                self.scheduler = DDIMScheduler.from_pretrained(self.pruned_model_path, subfolder="scheduler")
            
            self.model = self.pipeline.unet
            print("✓ Loaded as pipeline")
            
        except Exception as e:
            # Try to load direct model
            print(f"Pipeline loading failed ({e}), trying direct model loading...")
            
            # Look for pruned model file
            pruned_dir = os.path.join(self.pruned_model_path, "pruned")
            if os.path.exists(pruned_dir):
                model_files = [f for f in os.listdir(pruned_dir) if f.endswith('.pth')]
                if model_files:
                    model_file = os.path.join(pruned_dir, model_files[0])
                    self.model = torch.load(model_file, map_location='cpu')
                    print(f"✓ Loaded direct model: {model_files[0]}")
                    
                    # Create default scheduler
                    self.scheduler = DDPMScheduler(
                        num_train_timesteps=1000,
                        beta_start=0.0001,
                        beta_end=0.02,
                        beta_schedule="linear"
                    )
                    print("✓ Created default DDPM scheduler")
                else:
                    raise FileNotFoundError("No .pth files found in pruned directory")
            else:
                raise FileNotFoundError(f"No pruned directory found in {self.pruned_model_path}")
        
        self.model = self.model.to(self.device)
        self.model.eval()
        
        # Print model info
        total_params = sum(p.numel() for p in self.model.parameters())
        print(f"Model loaded: {total_params:,} parameters")
        
    def load_initial_masks(self) -> Dict[str, torch.Tensor]:
        """Load initial masks from disk."""
        print(f"Loading initial masks from: {self.initial_masks_dir}")
        
        if not os.path.exists(self.initial_masks_dir):
            raise FileNotFoundError(f"Masks directory not found: {self.initial_masks_dir}")
        
        mask_files = [f for f in os.listdir(self.initial_masks_dir) if f.endswith('.pt')]
        if not mask_files:
            raise FileNotFoundError(f"No .pt mask files found in {self.initial_masks_dir}")
        
        initial_masks = {}
        for mask_file in mask_files:
            # Convert filename back to layer name
            layer_name = mask_file.replace('.pt', '').replace('_', '.')
            mask_path = os.path.join(self.initial_masks_dir, mask_file)
            
            mask = torch.load(mask_path, map_location='cpu')
            initial_masks[layer_name] = mask
            
        print(f"✓ Loaded {len(initial_masks)} initial masks")
        return initial_masks
    
    def apply_initial_masks_to_model(self, initial_masks: Dict[str, torch.Tensor]):
        """Apply initial masks to the model using MaskPro wrappers."""
        print("Applying initial masks to model...")
        
        # Wrap model with MaskPro layers
        wrap_log = wrap_model_with_maskpro(
            self.model,
            n=self.n,
            m=self.m,
            target_layers=[r".*"],  # Match all layers
            exclude_layers=[r".*norm.*", r".*embed.*"]
        )
        
        wrapped_layers = sum(1 for status in wrap_log.values() if "✓" in status)
        print(f"✓ Wrapped {wrapped_layers} layers with MaskPro")
        
        # Set initial masks
        masks_applied = 0
        for name, module in self.model.named_modules():
            if hasattr(module, 'mask_logits'):
                # Try to find corresponding initial mask
                mask_applied = False
                for mask_name, mask in initial_masks.items():
                    if mask_name in name or name in mask_name:
                        try:
                            # Set the current mask directly
                            module.current_mask.copy_(mask.to(self.device))
                            
                            # Initialize logits to match the mask pattern
                            with torch.no_grad():
                                # This is a simplified initialization
                                # In practice, you might want more sophisticated initialization
                                module.mask_logits.data.fill_(0.0)
                            
                            masks_applied += 1
                            mask_applied = True
                            break
                        except Exception as e:
                            print(f"Warning: Failed to apply mask to {name}: {e}")
                
                if not mask_applied:
                    print(f"Warning: No initial mask found for layer {name}")
        
        print(f"✓ Applied initial masks to {masks_applied} layers")
        
    def get_dataset(self, dataset_name: str, dataset_size: int):
        """Get dataset for baseline computation."""
        print(f"Loading dataset: {dataset_name}")
        
        if dataset_name == "cifar10":
            dataset = utils.get_dataset("cifar10")
        elif dataset_name == "celeba":
            dataset = utils.get_dataset("celeba")
        else:
            raise ValueError(f"Unsupported dataset: {dataset_name}")
        
        # Limit dataset size
        if len(dataset) > dataset_size:
            indices = torch.randperm(len(dataset))[:dataset_size]
            dataset = torch.utils.data.Subset(dataset, indices)
        
        print(f"✓ Dataset loaded: {len(dataset)} samples")
        return dataset
    
    def sample_timesteps(self, batch_size: int, strategy: str = "uniform", 
                        subset: Optional[int] = None) -> torch.Tensor:
        """Sample timesteps for loss computation."""
        num_timesteps = self.scheduler.config.num_train_timesteps
        
        if subset is not None:
            num_timesteps = min(num_timesteps, subset)
        
        if strategy == "uniform":
            timesteps = torch.randint(0, num_timesteps, (batch_size,))
        elif strategy == "importance":
            # Sample more from earlier timesteps (higher noise)
            weights = torch.linspace(2.0, 0.5, num_timesteps)
            timesteps = torch.multinomial(weights, batch_size, replacement=True)
        elif strategy == "early":
            # Focus on early timesteps (high noise)
            timesteps = torch.randint(num_timesteps//2, num_timesteps, (batch_size,))
        elif strategy == "late":
            # Focus on late timesteps (low noise)
            timesteps = torch.randint(0, num_timesteps//2, (batch_size,))
        else:
            raise ValueError(f"Unknown timestep sampling strategy: {strategy}")
        
        return timesteps.to(self.device)
    
    def compute_diffusion_loss(self, batch: torch.Tensor, timesteps: torch.Tensor) -> torch.Tensor:
        """Compute diffusion denoising loss."""
        batch = batch.to(self.device)
        
        # Sample noise
        noise = torch.randn_like(batch)
        
        # Add noise to batch according to timesteps
        noisy_batch = self.scheduler.add_noise(batch, noise, timesteps)
        
        # Predict noise with model
        if self.mixed_precision:
            with torch.cuda.amp.autocast():
                noise_pred = self.model(noisy_batch, timesteps).sample
                loss = F.mse_loss(noise_pred, noise, reduction='mean')
        else:
            noise_pred = self.model(noisy_batch, timesteps).sample
            loss = F.mse_loss(noise_pred, noise, reduction='mean')
        
        return loss
    
    def compute_baseline_losses(self, 
                              dataset: torch.utils.data.Dataset,
                              batch_size: int,
                              timestep_sampling: str = "uniform",
                              timestep_subset: Optional[int] = None) -> List[float]:
        """Compute baseline losses across the dataset."""
        print("Computing baseline losses...")
        
        dataloader = torch.utils.data.DataLoader(
            dataset, 
            batch_size=batch_size, 
            shuffle=False,
            num_workers=2
        )
        
        baseline_losses = []
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(tqdm(dataloader, desc="Computing baseline")):
                # Handle different dataset formats
                if isinstance(batch, (list, tuple)):
                    batch = batch[0]  # Take images only
                
                current_batch_size = batch.size(0)
                
                # Sample timesteps for this batch
                timesteps = self.sample_timesteps(
                    current_batch_size, 
                    timestep_sampling,
                    timestep_subset
                )
                
                # Compute loss
                try:
                    loss = self.compute_diffusion_loss(batch, timesteps)
                    baseline_losses.append(loss.item())
                    
                except Exception as e:
                    print(f"Warning: Batch {batch_idx} failed: {e}")
                    continue
        
        if not baseline_losses:
            raise RuntimeError("No valid losses computed")
        
        print(f"✓ Computed {len(baseline_losses)} baseline losses")
        print(f"  Mean loss: {np.mean(baseline_losses):.6f}")
        print(f"  Std loss: {np.std(baseline_losses):.6f}")
        
        return baseline_losses
    
    def save_baseline_losses(self, losses: List[float], output_file: str, args):
        """Save baseline losses to file."""
        output_data = {
            'baseline_losses': losses,
            'statistics': {
                'mean': float(np.mean(losses)),
                'std': float(np.std(losses)),
                'min': float(np.min(losses)),
                'max': float(np.max(losses)),
                'count': len(losses)
            },
            'configuration': {
                'pruned_model_path': args.pruned_model_path,
                'initial_masks_dir': args.initial_masks_dir,
                'dataset': args.dataset,
                'dataset_size': args.dataset_size,
                'batch_size': args.batch_size,
                'num_train_timesteps': args.num_train_timesteps,
                'timestep_sampling': args.timestep_sampling,
                'timestep_subset': args.timestep_subset,
                'n': args.n,
                'm': args.m
            }
        }
        
        # Save as numpy array (for compatibility with original MaskPro)
        losses_array = np.array(losses)
        np.save(output_file.replace('.json', '.npy'), losses_array)
        
        # Also save as JSON for human readability
        with open(output_file, 'w') as f:
            json.dump(output_data, f, indent=2)
        
        print(f"✓ Saved baseline losses to: {output_file}")
        print(f"✓ Saved baseline array to: {output_file.replace('.json', '.npy')}")


def main():
    """Main baseline computation process."""
    args = parse_args()
    
    print("=" * 60)
    print("📊 Diffusion MaskPro: Baseline Loss Computation")
    print("=" * 60)
    
    # Set default output file if not provided
    if args.output_file is None:
        args.output_file = f"run/maskpro/baseline_losses_{args.dataset}_n{args.n}_m{args.m}.json"
        os.makedirs(os.path.dirname(args.output_file), exist_ok=True)
    
    try:
        # Create baseline computer
        computer = DiffusionBaselineLossComputer(
            pruned_model_path=args.pruned_model_path,
            initial_masks_dir=args.initial_masks_dir,
            n=args.n,
            m=args.m,
            device=args.device,
            mixed_precision=args.mixed_precision
        )
        
        # Load model and scheduler
        computer.load_model_and_scheduler()
        
        # Load initial masks
        initial_masks = computer.load_initial_masks()
        
        # Apply masks to model
        computer.apply_initial_masks_to_model(initial_masks)
        
        # Get dataset
        dataset = computer.get_dataset(args.dataset, args.dataset_size)
        
        # Compute baseline losses
        baseline_losses = computer.compute_baseline_losses(
            dataset=dataset,
            batch_size=args.batch_size,
            timestep_sampling=args.timestep_sampling,
            timestep_subset=args.timestep_subset
        )
        
        # Save results
        computer.save_baseline_losses(baseline_losses, args.output_file, args)
        
        print("\n✅ Baseline loss computation completed successfully!")
        
    except Exception as e:
        print(f"❌ Baseline computation failed: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code) 