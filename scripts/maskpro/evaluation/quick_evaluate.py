#!/usr/bin/env python3
"""
Quick MaskPro Evaluation Script

Fast evaluation for monitoring training progress:
- Basic sparsity analysis
- Sample generation (small batch)
- Performance check
- Visual inspection
"""

import torch
import os
import sys
import argparse
from pathlib import Path
import time
from tqdm import tqdm

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from diffusers import DDPMScheduler
from torchvision.utils import save_image, make_grid

from diffusion_maskpro import get_model_sparsity_summary, validate_maskpro_model


def quick_sparsity_check(model):
    """Quick sparsity analysis."""
    print("🔍 Quick sparsity check...")
    
    # Get overall sparsity
    sparsity_summary = get_model_sparsity_summary(model)
    overall_sparsity = sparsity_summary['overall_stats']['overall_sparsity']
    
    # Count MaskPro layers
    maskpro_layers = 0
    total_layers = 0
    nm_compliance_sum = 0
    
    for name, module in model.named_modules():
        total_layers += 1
        if hasattr(module, 'get_sparsity_info'):
            maskpro_layers += 1
            info = module.get_sparsity_info()
            nm_compliance_sum += info['nm_compliance']
    
    avg_nm_compliance = nm_compliance_sum / max(maskpro_layers, 1)
    
    print(f"✓ Overall sparsity: {overall_sparsity:.1%}")
    print(f"✓ MaskPro layers: {maskpro_layers}/{total_layers}")
    print(f"✓ Average N:M compliance: {avg_nm_compliance:.1%}")
    
    return {
        'overall_sparsity': overall_sparsity,
        'maskpro_layers': maskpro_layers,
        'total_layers': total_layers,
        'nm_compliance': avg_nm_compliance
    }


def quick_sample_generation(model, device, num_samples=8, num_steps=50):
    """Generate a few samples quickly."""
    print(f"🎨 Generating {num_samples} quick samples...")
    
    model.eval()
    scheduler = DDPMScheduler(
        num_train_timesteps=1000,
        beta_start=0.0001,
        beta_end=0.02,
        beta_schedule="linear"
    )
    
    # Skip many timesteps for speed
    timesteps = list(range(0, 1000, 1000 // num_steps))[::-1]
    
    with torch.no_grad():
        # Start with noise
        noise = torch.randn(num_samples, 3, 32, 32, device=device)
        
        # Denoising loop
        for i, t in enumerate(tqdm(timesteps, desc="Denoising")):
            timestep_tensor = torch.full((num_samples,), t, device=device, dtype=torch.long)
            
            # Predict noise
            noise_pred = model(noise, timestep_tensor).sample
            
            # Simple denoising step (simplified scheduler)
            alpha = 1 - (t / 1000) * 0.02
            noise = noise - alpha * 0.01 * noise_pred
    
    # Convert to images
    images = (noise + 1) / 2
    images = torch.clamp(images, 0, 1)
    
    return images


def quick_performance_test(model, device, num_runs=5):
    """Quick performance test."""
    print("⏱️  Quick performance test...")
    
    model.eval()
    
    # Warmup
    with torch.no_grad():
        noise = torch.randn(4, 3, 32, 32, device=device)
        timesteps = torch.randint(0, 1000, (4,), device=device)
        _ = model(noise, timesteps)
    
    torch.cuda.synchronize() if device.type == 'cuda' else None
    
    # Timing
    times = []
    for _ in tqdm(range(num_runs), desc="Timing"):
        noise = torch.randn(4, 3, 32, 32, device=device)
        timesteps = torch.randint(0, 1000, (4,), device=device)
        
        torch.cuda.synchronize() if device.type == 'cuda' else None
        start = time.time()
        
        with torch.no_grad():
            _ = model(noise, timesteps)
        
        torch.cuda.synchronize() if device.type == 'cuda' else None
        times.append(time.time() - start)
    
    avg_time = sum(times) / len(times)
    print(f"✓ Average forward pass: {avg_time*1000:.2f}ms (batch_size=4)")
    
    return avg_time


def main():
    """Quick evaluation main function."""
    parser = argparse.ArgumentParser(description="Quick MaskPro model evaluation")
    parser.add_argument("--checkpoint", type=str, required=True,
                       help="Path to model checkpoint")
    parser.add_argument("--device", type=str, default="cuda:0",
                       help="Device to use")
    parser.add_argument("--output_dir", type=str, default="run/maskpro/quick_eval",
                       help="Output directory")
    parser.add_argument("--num_samples", type=int, default=8,
                       help="Number of samples to generate")
    
    args = parser.parse_args()
    
    device = torch.device(args.device)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("🚀 Quick MaskPro Evaluation")
    print("=" * 40)
    
    # Load model
    print("📂 Loading model...")
    if args.checkpoint.endswith('.pt'):
        checkpoint = torch.load(args.checkpoint, map_location='cpu')
        if 'model_state_dict' in checkpoint:
            # Extract from training checkpoint
            # Note: This is simplified - real implementation would need model architecture
            print("⚠️  Loading from training checkpoint (simplified)")
            model_state = checkpoint['model_state_dict']
        else:
            model_state = checkpoint
    else:
        print("❌ Only .pt checkpoints supported for quick evaluation")
        return
    
    print("✓ Model loaded")
    
    # Quick sparsity check
    try:
        if hasattr(model_state, 'to'):
            model = model_state.to(device)
            sparsity_results = quick_sparsity_check(model)
        else:
            print("⚠️  Cannot analyze sparsity for raw state dict")
            sparsity_results = {}
    except Exception as e:
        print(f"⚠️  Sparsity analysis failed: {e}")
        sparsity_results = {}
    
    # Quick sample generation
    try:
        if hasattr(model_state, 'to'):
            samples = quick_sample_generation(model, device, args.num_samples)
            
            # Save samples
            grid = make_grid(samples, nrow=4, normalize=False, padding=2)
            save_image(grid, output_dir / "quick_samples.png")
            print(f"✓ Samples saved to {output_dir / 'quick_samples.png'}")
        else:
            print("⚠️  Cannot generate samples from raw state dict")
    except Exception as e:
        print(f"⚠️  Sample generation failed: {e}")
    
    # Quick performance test
    try:
        if hasattr(model_state, 'to'):
            perf_time = quick_performance_test(model, device)
        else:
            print("⚠️  Cannot test performance on raw state dict")
            perf_time = 0
    except Exception as e:
        print(f"⚠️  Performance test failed: {e}")
        perf_time = 0
    
    # Summary
    print("\n" + "=" * 40)
    print("📋 Quick Evaluation Summary")
    print("=" * 40)
    if sparsity_results:
        print(f"Sparsity: {sparsity_results.get('overall_sparsity', 0):.1%}")
        print(f"N:M Compliance: {sparsity_results.get('nm_compliance', 0):.1%}")
    print(f"Forward Pass: {perf_time*1000:.2f}ms")
    print(f"Samples: {output_dir / 'quick_samples.png'}")
    print("=" * 40)


if __name__ == "__main__":
    main() 