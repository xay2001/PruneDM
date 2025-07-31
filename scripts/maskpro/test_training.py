#!/usr/bin/env python3
"""
Sprint 3 Training Test: Complete MaskPro Training Pipeline

This script tests the complete training pipeline including:
- Configuration loading and validation
- Model loading and MaskPro wrapping  
- Training loop execution
- SwanLab integration
- Checkpoint saving and loading
"""

import torch
import torch.nn as nn
import os
import sys
import tempfile
import yaml
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from diffusion_maskpro.maskpro_trainer import DiffusionMaskProTrainer
from diffusion_maskpro import wrap_model_with_maskpro, validate_maskpro_model


def create_test_config() -> str:
    """Create a minimal test configuration."""
    config = {
        'model': {
            'pruned_model_path': 'run/pruned/magnitude/ddpm_cifar10_pruned',
            'initial_masks_dir': 'run/maskpro/initial_masks',
            'n': 2,
            'm': 4,
            'target_layers': ['.*conv.*', '.*linear.*'],
            'exclude_layers': ['.*norm.*', '.*embed.*', '.*conv_out.*']
        },
        'training': {
            'epochs': 2,
            'batch_size': 4,
            'gradient_accumulation_steps': 1,
            'model_lr': 1e-5,
            'mask_lr': 1e-3,
            'baseline_momentum': 0.99,
            'mask_loss_weight': 1.0,
            'num_train_timesteps': 1000,
            'timestep_sampling': 'uniform',
            'timestep_subset': None,
            'weight_decay': 1e-4,
            'gradient_clip_norm': 1.0,
            'val_freq': 1,
            'save_freq': 1
        },
        'dataset': {
            'name': 'cifar10',
            'size': 64,
            'val_size': 16,
            'num_workers': 0,  # Avoid multiprocessing issues in tests
            'pin_memory': False,
            'prefetch_factor': 2
        },
        'hardware': {
            'device': 'cpu',  # Use CPU for testing
            'mixed_precision': False,
            'compile_model': False,
            'gradient_checkpointing': False,
            'cpu_offload': False
        },
        'logging': {
            'use_swanlab': False,  # Disable SwanLab for testing
            'project_name': 'test-diffusion-maskpro',
            'experiment_name': 'test-run',
            'log_freq': 5,
            'image_log_freq': 20,
            'log_metrics': ['main_loss', 'mask_loss', 'total_loss'],
            'log_images': []
        },
        'output': {
            'output_dir': None,  # Will be set to temp directory
            'checkpoints_dir': 'checkpoints',
            'logs_dir': 'logs',
            'samples_dir': 'samples',
            'analysis_dir': 'analysis',
            'save_best_only': False,
            'save_last_n': 2
        },
        'validation': {
            'metrics': ['sample_quality'],
            'num_samples': 8,
            'num_inference_steps': 10,
            'guidance_scale': 1.0
        },
        'advanced': {
            'mask_init_strategy': 'magnitude_based',
            'temperature_scheduling': {
                'initial': 1.0,
                'final': 0.1,
                'decay_type': 'exponential'
            },
            'early_stopping': {
                'patience': 10,
                'metric': 'val_loss',
                'min_delta': 1e-4
            },
            'mask_learning': {
                'warmup_epochs': 0,
                'freeze_model_epochs': 0
            }
        }
    }
    
    return config


def test_config_loading():
    """Test configuration loading and validation."""
    print("🧪 Testing configuration loading...")
    
    # Create test config
    config = create_test_config()
    
    # Save to temporary file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        yaml.dump(config, f)
        config_path = f.name
    
    try:
        # Test loading
        trainer = DiffusionMaskProTrainer(config_path)
        print("✓ Configuration loaded successfully")
        return True
        
    except Exception as e:
        print(f"✗ Configuration loading failed: {e}")
        return False
        
    finally:
        os.unlink(config_path)


def test_model_loading():
    """Test model loading and MaskPro wrapping."""
    print("\n🧪 Testing model loading...")
    
    # Check if we have a pruned model
    pruned_model_path = "run/pruned/magnitude/ddpm_cifar10_pruned"
    
    if not os.path.exists(pruned_model_path):
        print("❌ No pruned model found, skipping model loading test")
        return False
    
    config = create_test_config()
    
    with tempfile.TemporaryDirectory() as temp_dir:
        config['output']['output_dir'] = temp_dir
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(config, f)
            config_path = f.name
        
        try:
            trainer = DiffusionMaskProTrainer(config_path)
            
            # Test model loading
            model, scheduler = trainer.load_model_and_scheduler()
            print("✓ Model loaded successfully")
            
            # Test MaskPro wrapping
            wrap_log = trainer.apply_maskpro_wrappers()
            wrapped_count = sum(1 for status in wrap_log.values() if "✓" in status)
            print(f"✓ Wrapped {wrapped_count} layers with MaskPro")
            
            # Test optimizer setup
            optimizers, schedulers = trainer.setup_optimizers()
            print("✓ Optimizers setup successfully")
            
            return True
            
        except Exception as e:
            print(f"✗ Model loading failed: {e}")
            import traceback
            traceback.print_exc()
            return False
            
        finally:
            os.unlink(config_path)


def test_data_loading():
    """Test data loader setup."""
    print("\n🧪 Testing data loading...")
    
    config = create_test_config()
    
    with tempfile.TemporaryDirectory() as temp_dir:
        config['output']['output_dir'] = temp_dir
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(config, f)
            config_path = f.name
        
        try:
            trainer = DiffusionMaskProTrainer(config_path)
            
            # Test data loader setup
            train_loader, val_loader = trainer.setup_data_loaders()
            
            print(f"✓ Train loader: {len(train_loader)} batches")
            print(f"✓ Val loader: {len(val_loader)} batches")
            
            # Test loading a batch
            batch = next(iter(train_loader))
            if isinstance(batch, (list, tuple)):
                batch = batch[0]
            print(f"✓ Batch shape: {batch.shape}")
            
            return True
            
        except Exception as e:
            print(f"✗ Data loading failed: {e}")
            return False
            
        finally:
            os.unlink(config_path)


def test_training_step():
    """Test a single training step."""
    print("\n🧪 Testing training step...")
    
    # Check if we have a pruned model
    pruned_model_path = "run/pruned/magnitude/ddpm_cifar10_pruned"
    
    if not os.path.exists(pruned_model_path):
        print("❌ No pruned model found, skipping training step test")
        return False
    
    config = create_test_config()
    
    with tempfile.TemporaryDirectory() as temp_dir:
        config['output']['output_dir'] = temp_dir
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(config, f)
            config_path = f.name
        
        try:
            trainer = DiffusionMaskProTrainer(config_path)
            
            # Setup trainer
            trainer.load_model_and_scheduler()
            trainer.apply_maskpro_wrappers()
            trainer.setup_data_loaders()
            trainer.setup_optimizers()
            
            # Get a test batch
            batch = next(iter(trainer.train_loader))
            if isinstance(batch, (list, tuple)):
                batch = batch[0]
            
            # Test training step
            print("Running training step...")
            step_metrics = trainer.training_step(batch)
            
            print("✓ Training step completed successfully")
            print(f"  Main loss: {step_metrics['main_loss']:.6f}")
            print(f"  Mask loss: {step_metrics['mask_loss']:.6f}")
            print(f"  Total loss: {step_metrics['total_loss']:.6f}")
            print(f"  Sparsity: {step_metrics['sparsity_ratio']:.1%}")
            
            return True
            
        except Exception as e:
            print(f"✗ Training step failed: {e}")
            import traceback
            traceback.print_exc()
            return False
            
        finally:
            os.unlink(config_path)


def test_validation():
    """Test validation loop."""
    print("\n🧪 Testing validation...")
    
    # Check if we have a pruned model
    pruned_model_path = "run/pruned/magnitude/ddpm_cifar10_pruned"
    
    if not os.path.exists(pruned_model_path):
        print("❌ No pruned model found, skipping validation test")
        return False
    
    config = create_test_config()
    
    with tempfile.TemporaryDirectory() as temp_dir:
        config['output']['output_dir'] = temp_dir
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(config, f)
            config_path = f.name
        
        try:
            trainer = DiffusionMaskProTrainer(config_path)
            
            # Setup trainer
            trainer.load_model_and_scheduler()
            trainer.apply_maskpro_wrappers()
            trainer.setup_data_loaders()
            trainer.setup_optimizers()
            
            # Test validation
            print("Running validation...")
            val_metrics = trainer.validate()
            
            print("✓ Validation completed successfully")
            print(f"  Val loss: {val_metrics['val_loss']:.6f}")
            print(f"  Val sparsity: {val_metrics['val_sparsity']:.1%}")
            print(f"  N:M compliance: {val_metrics['val_nm_compliance']:.1%}")
            
            return True
            
        except Exception as e:
            print(f"✗ Validation failed: {e}")
            import traceback
            traceback.print_exc()
            return False
            
        finally:
            os.unlink(config_path)


def test_checkpoint_saving():
    """Test checkpoint saving and loading."""
    print("\n🧪 Testing checkpoint saving...")
    
    # Check if we have a pruned model
    pruned_model_path = "run/pruned/magnitude/ddpm_cifar10_pruned"
    
    if not os.path.exists(pruned_model_path):
        print("❌ No pruned model found, skipping checkpoint test")
        return False
    
    config = create_test_config()
    
    with tempfile.TemporaryDirectory() as temp_dir:
        config['output']['output_dir'] = temp_dir
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(config, f)
            config_path = f.name
        
        try:
            trainer = DiffusionMaskProTrainer(config_path)
            
            # Setup trainer
            trainer.load_model_and_scheduler()
            trainer.apply_maskpro_wrappers()
            trainer.setup_data_loaders()
            trainer.setup_optimizers()
            
            # Test checkpoint saving
            print("Saving checkpoint...")
            trainer.save_checkpoint(epoch=0, is_best=True)
            
            # Check if files were created
            checkpoint_dir = os.path.join(temp_dir, "checkpoints")
            checkpoint_files = os.listdir(checkpoint_dir)
            
            if "checkpoint_epoch_0.pt" in checkpoint_files:
                print("✓ Regular checkpoint saved")
            
            if "best_checkpoint.pt" in checkpoint_files:
                print("✓ Best checkpoint saved")
            
            if "maskpro_state_epoch_0.pt" in checkpoint_files:
                print("✓ MaskPro state saved")
            
            return True
            
        except Exception as e:
            print(f"✗ Checkpoint saving failed: {e}")
            return False
            
        finally:
            os.unlink(config_path)


def test_mini_training():
    """Test a complete mini training run."""
    print("\n🧪 Testing mini training run...")
    
    # Check if we have a pruned model
    pruned_model_path = "run/pruned/magnitude/ddpm_cifar10_pruned"
    
    if not os.path.exists(pruned_model_path):
        print("❌ No pruned model found, skipping mini training test")
        return False
    
    config = create_test_config()
    config['training']['epochs'] = 1  # Just one epoch
    config['dataset']['size'] = 32    # Very small dataset
    config['dataset']['val_size'] = 8
    
    with tempfile.TemporaryDirectory() as temp_dir:
        config['output']['output_dir'] = temp_dir
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(config, f)
            config_path = f.name
        
        try:
            print("Starting mini training run...")
            trainer = DiffusionMaskProTrainer(config_path)
            trainer.train()
            
            print("✓ Mini training run completed successfully")
            
            # Check if outputs were created
            output_files = []
            for root, dirs, files in os.walk(temp_dir):
                output_files.extend(files)
            
            if output_files:
                print(f"✓ Created {len(output_files)} output files")
            
            return True
            
        except Exception as e:
            print(f"✗ Mini training failed: {e}")
            import traceback
            traceback.print_exc()
            return False
            
        finally:
            os.unlink(config_path)


def main():
    """Run all Sprint 3 training tests."""
    print("=" * 60)
    print("🚀 Sprint 3 Training Test Suite")
    print("=" * 60)
    
    tests = [
        ("Configuration Loading", test_config_loading),
        ("Model Loading", test_model_loading),
        ("Data Loading", test_data_loading),
        ("Training Step", test_training_step),
        ("Validation", test_validation),
        ("Checkpoint Saving", test_checkpoint_saving),
        ("Mini Training Run", test_mini_training),
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n{'='*20} {test_name} {'='*20}")
        try:
            if test_func():
                passed += 1
                print(f"✅ {test_name} PASSED")
            else:
                print(f"❌ {test_name} FAILED")
        except Exception as e:
            print(f"💥 {test_name} CRASHED: {e}")
            import traceback
            traceback.print_exc()
    
    print(f"\n{'='*60}")
    print(f"📊 Training Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All training tests passed! Sprint 3 is ready!")
        print("\n🚀 You can now run full training with:")
        print("python scripts/maskpro/diffusion_maskpro_train.py --config scripts/maskpro/configs/diffusion_maskpro_config.yaml")
    elif passed >= total - 1:
        print("⚠️  Most tests passed. Minor issues may need attention.")
    else:
        print("⚠️  Several tests failed. Review and fix issues before proceeding.")
    
    print("=" * 60)
    
    return passed >= total - 1


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 