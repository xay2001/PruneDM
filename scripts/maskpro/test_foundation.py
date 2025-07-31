#!/usr/bin/env python3
"""
Foundation Layer Test Script for Diffusion MaskPro

This script tests the core components of the MaskPro foundation layer
to ensure everything is working correctly before proceeding to integration.
"""

import torch
import torch.nn as nn
import sys
import os

# Add the project root to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

from diffusion_maskpro import (
    MaskProLayer,
    validate_nm_compatibility,
    suggest_nm_pattern,
    wrap_model_with_maskpro,
    get_model_sparsity_summary,
    validate_maskpro_model
)


def test_maskpro_layer():
    """Test basic MaskProLayer functionality."""
    print("🧪 Testing MaskProLayer...")
    
    # Test with Conv2d layer
    conv_layer = nn.Conv2d(64, 128, kernel_size=3, padding=1)
    print(f"Original Conv2d shape: {conv_layer.weight.shape}")
    
    try:
        # Create MaskPro wrapper
        maskpro_conv = MaskProLayer(conv_layer, n=2, m=4)
        print(f"✓ MaskPro Conv2d wrapper created successfully")
        print(f"  - Original shape: {maskpro_conv.original_weight_shape}")
        print(f"  - N:M pattern: {maskpro_conv.n}:{maskpro_conv.m}")
        print(f"  - Total groups: {maskpro_conv.num_groups}")
        
        # Test forward pass
        dummy_input = torch.randn(2, 64, 32, 32)
        output = maskpro_conv(dummy_input)
        print(f"✓ Forward pass successful, output shape: {output.shape}")
        
        # Test mask generation
        mask = maskpro_conv.generate_nm_mask(training=False)
        print(f"✓ Mask generation successful, mask shape: {mask.shape}")
        
        # Test sparsity info
        sparsity_info = maskpro_conv.get_sparsity_info()
        print(f"✓ Sparsity info: {sparsity_info['sparsity_ratio']:.1%} sparse")
        print(f"  - N:M compliance: {sparsity_info['nm_compliance']:.1%}")
        
    except Exception as e:
        print(f"✗ Conv2d test failed: {e}")
        return False
    
    # Test with Linear layer
    linear_layer = nn.Linear(512, 256)
    print(f"\nOriginal Linear shape: {linear_layer.weight.shape}")
    
    try:
        # Create MaskPro wrapper
        maskpro_linear = MaskProLayer(linear_layer, n=2, m=4)
        print(f"✓ MaskPro Linear wrapper created successfully")
        
        # Test forward pass
        dummy_input = torch.randn(8, 512)
        output = maskpro_linear(dummy_input)
        print(f"✓ Forward pass successful, output shape: {output.shape}")
        
        # Test policy gradient loss
        main_loss = torch.tensor(0.5)
        mask_loss = maskpro_linear.get_mask_loss(main_loss)
        print(f"✓ Policy gradient loss computed: {mask_loss.item():.4f}")
        
    except Exception as e:
        print(f"✗ Linear test failed: {e}")
        return False
    
    return True


def test_nm_compatibility():
    """Test N:M compatibility validation."""
    print("\n🧪 Testing N:M compatibility validation...")
    
    # Test compatible layers
    conv_64_128_3x3 = nn.Conv2d(64, 128, kernel_size=3)  # 64*3*3 = 576, divisible by 4
    linear_512_256 = nn.Linear(512, 256)  # 512 divisible by 4
    
    # Test incompatible layers  
    conv_odd = nn.Conv2d(63, 128, kernel_size=3)  # 63*3*3 = 567, not divisible by 4
    linear_odd = nn.Linear(511, 256)  # 511 not divisible by 4
    
    test_cases = [
        (conv_64_128_3x3, 2, 4, True, "Compatible Conv2d"),
        (linear_512_256, 2, 4, True, "Compatible Linear"),
        (conv_odd, 2, 4, False, "Incompatible Conv2d"),
        (linear_odd, 2, 4, False, "Incompatible Linear"),
        (conv_64_128_3x3, 1, 4, True, "1:4 sparsity"),
        (conv_64_128_3x3, 3, 4, True, "3:4 sparsity"),
    ]
    
    for layer, n, m, expected, description in test_cases:
        is_compatible, reason = validate_nm_compatibility(layer, n, m)
        
        if is_compatible == expected:
            status = "✓"
        else:
            status = "✗"
            
        print(f"{status} {description}: {is_compatible} - {reason}")
    
    return True


def test_pattern_suggestions():
    """Test N:M pattern suggestions."""
    print("\n🧪 Testing N:M pattern suggestions...")
    
    # Test different layer configurations
    layers = [
        nn.Conv2d(64, 128, kernel_size=3),  # 64*9 = 576 (divisible by many)
        nn.Linear(512, 256),                # 512 (power of 2)
        nn.Conv2d(32, 64, kernel_size=5),   # 32*25 = 800 (divisible by 4, 8)
        nn.Linear(1024, 512),               # 1024 (divisible by many)
    ]
    
    for i, layer in enumerate(layers):
        patterns = suggest_nm_pattern(layer)
        print(f"Layer {i+1} ({layer.__class__.__name__} {layer.weight.shape}):")
        for j, (n, m) in enumerate(patterns[:3]):  # Show top 3 suggestions
            print(f"  {j+1}. {n}:{m} sparsity")
    
    return True


def test_model_wrapping():
    """Test model wrapping functionality."""
    print("\n🧪 Testing model wrapping...")
    
    # Create a simple test model
    class SimpleModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
            self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
            self.norm = nn.BatchNorm2d(128)  # Should be excluded
            self.linear = nn.Linear(128, 10)
            self.final_conv = nn.Conv2d(128, 10, kernel_size=1)  # Might be excluded
            
        def forward(self, x):
            x = self.conv1(x)
            x = self.conv2(x)
            x = self.norm(x)
            x = torch.mean(x, dim=(2, 3))  # Global average pooling
            x = self.linear(x)
            return x
    
    model = SimpleModel()
    print(f"Original model parameters: {sum(p.numel() for p in model.parameters())}")
    
    # Wrap with MaskPro
    wrap_log = wrap_model_with_maskpro(model, n=2, m=4)
    
    print("Wrapping results:")
    for layer_name, status in wrap_log.items():
        if "✓" in status:
            print(f"  ✓ {layer_name}: {status}")
        elif "✗" in status:
            print(f"  ✗ {layer_name}: {status}")
        else:
            print(f"  - {layer_name}: {status}")
    
    # Validate wrapped model
    validation = validate_maskpro_model(model)
    print(f"\nModel validation: {'✓ Valid' if validation['is_valid'] else '✗ Invalid'}")
    print(f"MaskPro layers found: {validation['layer_count']}")
    print(f"Mask parameters: {validation['mask_param_count']}")
    
    if validation['errors']:
        for error in validation['errors']:
            print(f"  Error: {error}")
    
    # Test forward pass with wrapped model
    try:
        dummy_input = torch.randn(2, 3, 32, 32)
        output = model(dummy_input)
        print(f"✓ Wrapped model forward pass successful, output shape: {output.shape}")
    except Exception as e:
        print(f"✗ Wrapped model forward pass failed: {e}")
        return False
    
    # Get sparsity summary
    summary = get_model_sparsity_summary(model)
    print(f"\nSparsity summary:")
    print(f"  - MaskPro layers: {summary['overall_stats']['total_maskpro_layers']}")
    print(f"  - Overall sparsity: {summary['overall_stats']['overall_sparsity']:.1%}")
    print(f"  - N:M patterns: {summary['overall_stats']['nm_patterns']}")
    
    return True


def test_training_simulation():
    """Test a simple training-like scenario."""
    print("\n🧪 Testing training simulation...")
    
    # Create a simple model with N:M compatible layers
    model = nn.Sequential(
        nn.Conv2d(4, 32, kernel_size=3, padding=1),    # 4*3*3=36, divisible by 4
        nn.ReLU(),
        nn.Conv2d(32, 64, kernel_size=3, padding=1),   # 32*3*3=288, divisible by 4
        nn.ReLU(),
        nn.AdaptiveAvgPool2d((1, 1)),
        nn.Flatten(),
        nn.Linear(64, 10)                              # 64, divisible by 4
    )
    
    # Wrap with MaskPro - target all Conv2d and Linear layers regardless of name
    wrap_log = wrap_model_with_maskpro(
        model, 
        n=2, 
        m=4,
        target_layers=[r".*"],  # Match all layer names
        exclude_layers=[r".*relu.*", r".*pool.*", r".*flatten.*"]  # Exclude non-parameterized layers
    )
    wrapped_layers = sum(1 for status in wrap_log.values() if "✓" in status)
    print(f"Wrapped {wrapped_layers} layers successfully")
    
    # Simulate training setup
    dummy_input = torch.randn(4, 4, 32, 32)  # Changed to 4 channels to match model
    dummy_target = torch.randint(0, 10, (4,))
    
    # Create separate optimizers for model and mask parameters
    mask_params = [p for name, p in model.named_parameters() if 'mask_logits' in name]
    model_params = [p for name, p in model.named_parameters() if 'mask_logits' not in name]
    
    print(f"Model parameters: {len(model_params)}")
    print(f"Mask parameters: {len(mask_params)}")
    
    if len(mask_params) == 0:
        print("✗ No mask parameters found - wrapping may have failed")
        return False
    
    optimizer_model = torch.optim.Adam(model_params, lr=1e-3)
    optimizer_mask = torch.optim.Adam(mask_params, lr=1e-2)
    
    # Simulate a few training steps
    model.train()
    for step in range(3):
        # Forward pass
        output = model(dummy_input)
        main_loss = nn.CrossEntropyLoss()(output, dummy_target)
        
        # Update model parameters
        optimizer_model.zero_grad()
        main_loss.backward(retain_graph=True)
        optimizer_model.step()
        
        # Compute and update mask parameters
        total_mask_loss = torch.tensor(0.0)
        for module in model.modules():
            if hasattr(module, 'get_mask_loss'):
                mask_loss = module.get_mask_loss(main_loss)
                total_mask_loss += mask_loss
        
        if total_mask_loss.requires_grad:
            optimizer_mask.zero_grad()
            total_mask_loss.backward()
            optimizer_mask.step()
        
        print(f"  Step {step+1}: Main loss = {main_loss.item():.4f}, "
              f"Mask loss = {total_mask_loss.item():.4f}")
    
    print("✓ Training simulation completed successfully")
    return True


def main():
    """Run all foundation tests."""
    print("=" * 60)
    print("🚀 Diffusion MaskPro Foundation Layer Test Suite")
    print("=" * 60)
    
    tests = [
        ("MaskPro Layer", test_maskpro_layer),
        ("N:M Compatibility", test_nm_compatibility), 
        ("Pattern Suggestions", test_pattern_suggestions),
        ("Model Wrapping", test_model_wrapping),
        ("Training Simulation", test_training_simulation),
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
    print(f"📊 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All foundation tests passed! Ready for Sprint 2.")
    else:
        print("⚠️  Some tests failed. Please fix issues before proceeding.")
    
    print("=" * 60)
    
    return passed == total


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 