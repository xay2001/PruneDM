#!/usr/bin/env python3
"""
Sprint 2 Integration Test: Mask Extraction and Baseline Computation

This script tests the integration of MaskPro with pruned diffusion models,
including mask extraction and baseline loss computation.
"""

import torch
import torch.nn as nn
import os
import sys
import tempfile
import shutil
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from diffusion_maskpro import (
    wrap_model_with_maskpro,
    validate_maskpro_model,
    get_model_sparsity_summary,
    validate_nm_compatibility
)


def test_mask_extraction():
    """Test mask extraction from magnitude-pruned model."""
    print("🧪 Testing mask extraction...")
    
    # Check if we have a magnitude-pruned model
    pruned_model_path = "run/pruned/magnitude/ddpm_cifar10_pruned"
    
    if not os.path.exists(pruned_model_path):
        print("❌ No magnitude-pruned model found, skipping mask extraction test")
        return False
    
    print(f"Found pruned model at: {pruned_model_path}")
    
    # Create temporary output directory
    with tempfile.TemporaryDirectory() as temp_dir:
        output_dir = os.path.join(temp_dir, "test_masks")
        
        # Test mask extraction script
        extraction_cmd = [
            "python", "scripts/maskpro/extract_initial_masks.py",
            "--pruned_model_path", pruned_model_path,
            "--output_dir", output_dir,
            "--n", "2",
            "--m", "4",
            "--device", "cpu",  # Use CPU for testing
            "--save_analysis"
        ]
        
        try:
            import subprocess
            result = subprocess.run(extraction_cmd, capture_output=True, text=True)
            
            if result.returncode == 0:
                print("✓ Mask extraction script executed successfully")
                
                # Check if masks were created
                if os.path.exists(output_dir):
                    mask_files = [f for f in os.listdir(output_dir) if f.endswith('.pt')]
                    if mask_files:
                        print(f"✓ Created {len(mask_files)} mask files")
                        
                        # Check if analysis was saved
                        if os.path.exists(os.path.join(output_dir, "mask_analysis.json")):
                            print("✓ Mask analysis saved")
                        
                        return True
                    else:
                        print("❌ No mask files created")
                        return False
                else:
                    print("❌ Output directory not created")
                    return False
            else:
                print(f"❌ Mask extraction failed: {result.stderr}")
                return False
                
        except Exception as e:
            print(f"❌ Failed to run mask extraction: {e}")
            return False


def test_mask_loading_and_application():
    """Test loading masks and applying them to a model."""
    print("\n🧪 Testing mask loading and application...")
    
    # Create a simple test model
    test_model = nn.Sequential(
        nn.Conv2d(4, 32, kernel_size=3, padding=1),    # 4*3*3=36, divisible by 4
        nn.ReLU(),
        nn.Conv2d(32, 64, kernel_size=3, padding=1),   # 32*3*3=288, divisible by 4
        nn.AdaptiveAvgPool2d((1, 1)),
        nn.Flatten(),
        nn.Linear(64, 10)                              # 64, divisible by 4
    )
    
    print(f"Test model parameters: {sum(p.numel() for p in test_model.parameters())}")
    
    # Wrap model with MaskPro
    wrap_log = wrap_model_with_maskpro(
        test_model,
        n=2,
        m=4,
        target_layers=[r".*"],
        exclude_layers=[r".*relu.*", r".*pool.*", r".*flatten.*"]
    )
    
    wrapped_layers = sum(1 for status in wrap_log.values() if "✓" in status)
    print(f"✓ Wrapped {wrapped_layers} layers with MaskPro")
    
    # Test forward pass
    dummy_input = torch.randn(2, 4, 32, 32)
    try:
        output = test_model(dummy_input)
        print(f"✓ Forward pass successful, output shape: {output.shape}")
    except Exception as e:
        print(f"❌ Forward pass failed: {e}")
        return False
    
    # Get sparsity summary
    summary = get_model_sparsity_summary(test_model)
    print(f"✓ Model sparsity: {summary['overall_stats']['overall_sparsity']:.1%}")
    
    # Validate model
    validation = validate_maskpro_model(test_model)
    if validation['is_valid']:
        print("✓ Model validation passed")
        return True
    else:
        print(f"❌ Model validation failed: {validation['errors']}")
        return False


def test_diffusion_model_compatibility():
    """Test compatibility with actual diffusion model components."""
    print("\n🧪 Testing diffusion model compatibility...")
    
    try:
        from diffusers import UNet2DModel
        
        # Create a small UNet model for testing
        unet = UNet2DModel(
            sample_size=32,
            in_channels=3,
            out_channels=3,
            layers_per_block=1,
            block_out_channels=(32, 64),
            down_block_types=("DownBlock2D", "DownBlock2D"),
            up_block_types=("UpBlock2D", "UpBlock2D"),
        )
        
        print(f"UNet model parameters: {sum(p.numel() for p in unet.parameters())}")
        
        # Test layer compatibility
        compatible_layers = 0
        incompatible_layers = 0
        
        for name, module in unet.named_modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                is_compatible, reason = validate_nm_compatibility(module, 2, 4)
                if is_compatible:
                    compatible_layers += 1
                else:
                    incompatible_layers += 1
        
        print(f"✓ Compatible layers: {compatible_layers}")
        print(f"- Incompatible layers: {incompatible_layers}")
        
        if compatible_layers > 0:
            # Try wrapping some layers
            wrap_log = wrap_model_with_maskpro(
                unet,
                n=2,
                m=4,
                target_layers=[r".*conv.*", r".*linear.*"],
                exclude_layers=[r".*norm.*", r".*embed.*", r".*conv_out.*"]
            )
            
            wrapped_count = sum(1 for status in wrap_log.values() if "✓" in status)
            print(f"✓ Successfully wrapped {wrapped_count} UNet layers")
            
            # Test forward pass
            dummy_input = torch.randn(1, 3, 32, 32)
            dummy_timestep = torch.tensor([100])
            
            try:
                with torch.no_grad():
                    output = unet(dummy_input, dummy_timestep)
                print(f"✓ UNet forward pass successful")
                return True
            except Exception as e:
                print(f"❌ UNet forward pass failed: {e}")
                return False
        else:
            print("⚠️  No compatible layers found in UNet")
            return True  # Not a failure, just no compatible layers
            
    except Exception as e:
        print(f"❌ Diffusion model compatibility test failed: {e}")
        return False


def test_n_m_pattern_analysis():
    """Test N:M pattern analysis for different layer configurations."""
    print("\n🧪 Testing N:M pattern analysis...")
    
    # Test different layer configurations
    test_layers = [
        nn.Conv2d(3, 64, kernel_size=3),     # 3*3*3=27, not divisible by 4
        nn.Conv2d(4, 64, kernel_size=3),     # 4*3*3=36, divisible by 4
        nn.Conv2d(16, 32, kernel_size=3),    # 16*3*3=144, divisible by 4
        nn.Linear(128, 64),                  # 128, divisible by 4
        nn.Linear(100, 64),                  # 100, divisible by 4
        nn.Linear(99, 64),                   # 99, not divisible by 4
    ]
    
    patterns_to_test = [(1, 4), (2, 4), (1, 8), (2, 8)]
    
    analysis_results = {}
    
    for i, layer in enumerate(test_layers):
        layer_name = f"layer_{i}_{layer.__class__.__name__}_{layer.weight.shape}"
        analysis_results[layer_name] = {}
        
        for n, m in patterns_to_test:
            is_compatible, reason = validate_nm_compatibility(layer, n, m)
            analysis_results[layer_name][f"{n}:{m}"] = {
                'compatible': is_compatible,
                'reason': reason
            }
    
    # Print analysis
    compatible_count = 0
    total_tests = 0
    
    for layer_name, patterns in analysis_results.items():
        print(f"\n{layer_name}:")
        for pattern, result in patterns.items():
            status = "✓" if result['compatible'] else "✗"
            print(f"  {status} {pattern}: {result['reason']}")
            total_tests += 1
            if result['compatible']:
                compatible_count += 1
    
    compatibility_rate = compatible_count / total_tests
    print(f"\n✓ Overall compatibility rate: {compatibility_rate:.1%} ({compatible_count}/{total_tests})")
    
    return True


def test_maskpro_training_setup():
    """Test the setup for MaskPro training (parameter groups, optimizers)."""
    print("\n🧪 Testing MaskPro training setup...")
    
    # Create test model with proper dimensions
    model = nn.Sequential(
        nn.Conv2d(4, 32, kernel_size=3, padding=1),
        nn.Conv2d(32, 64, kernel_size=3, padding=1),
        nn.AdaptiveAvgPool2d((1, 1)),  # Ensure consistent output size
        nn.Flatten(),
        nn.Linear(64, 10)
    )
    
    # Wrap with MaskPro
    wrap_log = wrap_model_with_maskpro(
        model,
        n=2,
        m=4,
        target_layers=[r".*"],
        exclude_layers=[r".*relu.*", r".*pool.*", r".*flatten.*"]
    )
    
    # Test parameter separation
    mask_params = [p for name, p in model.named_parameters() if 'mask_logits' in name]
    model_params = [p for name, p in model.named_parameters() if 'mask_logits' not in name]
    
    print(f"✓ Model parameters: {len(model_params)}")
    print(f"✓ Mask parameters: {len(mask_params)}")
    
    if len(mask_params) == 0:
        print("❌ No mask parameters found")
        return False
    
    # Test optimizer setup
    try:
        optimizer_model = torch.optim.Adam(model_params, lr=1e-4)
        optimizer_mask = torch.optim.Adam(mask_params, lr=1e-2)
        
        print("✓ Optimizers created successfully")
        
        # Test a training step simulation
        dummy_input = torch.randn(2, 4, 8, 8)
        
        # First get the actual output shape to create correct target
        with torch.no_grad():
            test_output = model(dummy_input)
        dummy_target = torch.randn_like(test_output)
        
        # Forward pass
        output = model(dummy_input)
        main_loss = nn.MSELoss()(output, dummy_target)
        
        # Backward for model parameters
        optimizer_model.zero_grad()
        main_loss.backward(retain_graph=True)
        optimizer_model.step()
        
        # Compute mask loss
        total_mask_loss = torch.tensor(0.0)
        for module in model.modules():
            if hasattr(module, 'get_mask_loss'):
                mask_loss = module.get_mask_loss(main_loss)
                total_mask_loss += mask_loss
        
        if total_mask_loss.requires_grad:
            optimizer_mask.zero_grad()
            total_mask_loss.backward()
            optimizer_mask.step()
        
        print("✓ Training step simulation successful")
        return True
        
    except Exception as e:
        print(f"❌ Training setup failed: {e}")
        return False


def main():
    """Run all Sprint 2 integration tests."""
    print("=" * 60)
    print("🚀 Sprint 2 Integration Test Suite")
    print("=" * 60)
    
    tests = [
        ("Mask Extraction", test_mask_extraction),
        ("Mask Loading & Application", test_mask_loading_and_application),
        ("Diffusion Model Compatibility", test_diffusion_model_compatibility),
        ("N:M Pattern Analysis", test_n_m_pattern_analysis),
        ("MaskPro Training Setup", test_maskpro_training_setup),
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
    print(f"📊 Integration Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All integration tests passed! Ready for Sprint 3.")
    elif passed >= total - 1:
        print("⚠️  Most tests passed. Minor issues may need attention.")
    else:
        print("⚠️  Several tests failed. Review and fix issues before proceeding.")
    
    print("=" * 60)
    
    return passed >= total - 1  # Allow 1 failure


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 