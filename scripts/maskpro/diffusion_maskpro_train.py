#!/usr/bin/env python3
"""
Diffusion MaskPro Training Script

This script implements the complete two-stage hybrid pruning pipeline:
Stage 1: Magnitude pruning (already completed)
Stage 2: N:M sparsity learning using MaskPro

Usage:
    python scripts/maskpro/diffusion_maskpro_train.py --config configs/diffusion_maskpro_config.yaml
"""

import torch
import argparse
import sys
import os
import time
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from diffusion_maskpro.maskpro_trainer import DiffusionMaskProTrainer


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Train MaskPro on magnitude-pruned diffusion models")
    
    # Configuration
    parser.add_argument("--config", type=str, required=True,
                       help="Path to training configuration file")
    
    # Override options
    parser.add_argument("--device", type=str, default=None,
                       help="Override device (e.g., cuda:0, cpu)")
    parser.add_argument("--batch_size", type=int, default=None,
                       help="Override batch size")
    parser.add_argument("--epochs", type=int, default=None,
                       help="Override number of epochs")
    parser.add_argument("--output_dir", type=str, default=None,
                       help="Override output directory")
    
    # Logging options
    parser.add_argument("--experiment_name", type=str, default=None,
                       help="Override experiment name for logging")
    parser.add_argument("--no_swanlab", action="store_true",
                       help="Disable SwanLab logging")
    
    # Model options
    parser.add_argument("--pruned_model_path", type=str, default=None,
                       help="Override path to pruned model")
    parser.add_argument("--initial_masks_dir", type=str, default=None,
                       help="Override path to initial masks directory")
    
    # Quick test mode
    parser.add_argument("--quick_test", action="store_true",
                       help="Run in quick test mode (small dataset, few epochs)")
    
    return parser.parse_args()


def apply_config_overrides(config_path: str, args) -> str:
    """Apply command line overrides to configuration and save temporary config."""
    import yaml
    import tempfile
    
    # Load original config
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Apply overrides
    if args.device is not None:
        config['hardware']['device'] = args.device
    
    if args.batch_size is not None:
        config['training']['batch_size'] = args.batch_size
    
    if args.epochs is not None:
        config['training']['epochs'] = args.epochs
    
    if args.output_dir is not None:
        config['output']['output_dir'] = args.output_dir
    
    if args.experiment_name is not None:
        config['logging']['experiment_name'] = args.experiment_name
    
    if args.no_swanlab:
        config['logging']['use_swanlab'] = False
    
    if args.pruned_model_path is not None:
        config['model']['pruned_model_path'] = args.pruned_model_path
    
    if args.initial_masks_dir is not None:
        config['model']['initial_masks_dir'] = args.initial_masks_dir
    
    # Quick test mode overrides
    if args.quick_test:
        config['training']['epochs'] = 3
        config['training']['batch_size'] = 8
        config['dataset']['size'] = 256
        config['dataset']['val_size'] = 64
        config['logging']['log_freq'] = 5
        config['logging']['experiment_name'] += "_quick_test"
        print("🚀 Quick test mode enabled!")
    
    # Create temporary config file
    temp_config = tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False)
    yaml.dump(config, temp_config, default_flow_style=False)
    temp_config.close()
    
    return temp_config.name


def check_prerequisites(config_path: str):
    """Check if all prerequisites are met."""
    import yaml
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    print("🔍 Checking prerequisites...")
    
    # Check pruned model
    pruned_model_path = config['model']['pruned_model_path']
    if not os.path.exists(pruned_model_path):
        raise FileNotFoundError(f"Pruned model not found: {pruned_model_path}")
    
    # Check for pruned model file
    pruned_dir = os.path.join(pruned_model_path, "pruned")
    if os.path.exists(pruned_dir):
        model_files = [f for f in os.listdir(pruned_dir) if f.endswith('.pth')]
        if not model_files:
            raise FileNotFoundError(f"No .pth files found in {pruned_dir}")
        print(f"✓ Found pruned model: {model_files[0]}")
    else:
        print(f"✓ Will attempt pipeline loading from: {pruned_model_path}")
    
    # Check initial masks (optional)
    initial_masks_dir = config['model'].get('initial_masks_dir')
    if initial_masks_dir and os.path.exists(initial_masks_dir):
        mask_files = [f for f in os.listdir(initial_masks_dir) if f.endswith('.pt')]
        print(f"✓ Found {len(mask_files)} initial mask files")
    else:
        print("ℹ️  No initial masks directory found, will use random initialization")
    
    # Check CUDA availability if specified
    device = config['hardware']['device']
    if 'cuda' in device and not torch.cuda.is_available():
        print(f"⚠️  CUDA device {device} specified but CUDA not available, falling back to CPU")
        config['hardware']['device'] = 'cpu'
    
    # Check SwanLab if enabled
    if config['logging']['use_swanlab']:
        try:
            import swanlab
            print("✓ SwanLab available for logging")
        except ImportError:
            print("⚠️  SwanLab not available, logging will be limited")
            config['logging']['use_swanlab'] = False
    
    print("✅ Prerequisites check completed")


def print_training_info(config_path: str):
    """Print training configuration summary."""
    import yaml
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    print("\n" + "=" * 60)
    print("📋 DIFFUSION MASKPRO TRAINING CONFIGURATION")
    print("=" * 60)
    
    # Model info
    print(f"🎯 Model: {config['model']['pruned_model_path']}")
    print(f"   N:M Pattern: {config['model']['n']}:{config['model']['m']}")
    print(f"   Target Layers: {', '.join(config['model']['target_layers'])}")
    
    # Training info
    print(f"\n🚀 Training:")
    print(f"   Epochs: {config['training']['epochs']}")
    print(f"   Batch Size: {config['training']['batch_size']}")
    print(f"   Model LR: {config['training']['model_lr']}")
    print(f"   Mask LR: {config['training']['mask_lr']}")
    
    # Dataset info
    print(f"\n📊 Dataset:")
    print(f"   Name: {config['dataset']['name']}")
    print(f"   Train Size: {config['dataset']['size']}")
    print(f"   Val Size: {config['dataset']['val_size']}")
    
    # Hardware info
    print(f"\n💻 Hardware:")
    print(f"   Device: {config['hardware']['device']}")
    print(f"   Mixed Precision: {config['hardware']['mixed_precision']}")
    
    # Output info
    print(f"\n📁 Output:")
    print(f"   Directory: {config['output']['output_dir']}")
    if config['logging']['use_swanlab']:
        print(f"   SwanLab: {config['logging']['project_name']}/{config['logging']['experiment_name']}")
    
    print("=" * 60)


def estimate_training_time(config_path: str):
    """Estimate training time based on configuration."""
    import yaml
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Simple estimation based on dataset size and epochs
    dataset_size = config['dataset']['size']
    batch_size = config['training']['batch_size']
    epochs = config['training']['epochs']
    
    steps_per_epoch = dataset_size // batch_size
    total_steps = steps_per_epoch * epochs
    
    # Rough estimation: 0.5-2 seconds per step depending on hardware
    device = config['hardware']['device']
    if 'cuda' in device:
        time_per_step = 1.0  # seconds
    else:
        time_per_step = 3.0  # CPU is slower
    
    estimated_time_hours = (total_steps * time_per_step) / 3600
    
    print(f"\n⏱️  Estimated Training Time:")
    print(f"   Steps per epoch: {steps_per_epoch}")
    print(f"   Total steps: {total_steps}")
    print(f"   Estimated time: {estimated_time_hours:.1f} hours")
    
    if estimated_time_hours > 24:
        print("   ⚠️  Training will take more than 24 hours")
    elif estimated_time_hours > 8:
        print("   ⚠️  Training will take more than 8 hours")
    else:
        print("   ✓ Training time seems reasonable")


def main():
    """Main training function."""
    args = parse_args()
    
    print("🎭 Diffusion MaskPro Training")
    print(f"Configuration: {args.config}")
    
    try:
        # Check if config file exists
        if not os.path.exists(args.config):
            raise FileNotFoundError(f"Configuration file not found: {args.config}")
        
        # Apply command line overrides
        temp_config_path = apply_config_overrides(args.config, args)
        
        # Check prerequisites
        check_prerequisites(temp_config_path)
        
        # Print configuration summary
        print_training_info(temp_config_path)
        
        # Estimate training time
        estimate_training_time(temp_config_path)
        
        # Confirm before starting
        if not args.quick_test:
            confirm = input("\n🚀 Start training? [y/N]: ")
            if confirm.lower() not in ['y', 'yes']:
                print("Training cancelled.")
                return
        
        print(f"\n⏰ Training started at: {time.strftime('%Y-%m-%d %H:%M:%S')}")
        
        # Create trainer and start training
        trainer = DiffusionMaskProTrainer(temp_config_path)
        trainer.train()
        
        print(f"\n🎉 Training completed at: {time.strftime('%Y-%m-%d %H:%M:%S')}")
        
    except KeyboardInterrupt:
        print("\n⚠️  Training interrupted by user")
        return 1
        
    except Exception as e:
        print(f"\n❌ Training failed: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    finally:
        # Clean up temporary config file
        if 'temp_config_path' in locals() and os.path.exists(temp_config_path):
            os.unlink(temp_config_path)
    
    return 0


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code) 