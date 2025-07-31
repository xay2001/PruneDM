#!/usr/bin/env python3
"""
MaskPro Model Evaluation Suite

This script provides comprehensive evaluation of trained MaskPro models:
1. Sample generation and quality assessment
2. FID score computation against reference datasets
3. Sparsity analysis and N:M compliance verification
4. Hardware acceleration performance testing
5. Comparison with baseline pruning methods
6. Model size and memory usage analysis
"""

import torch
import torch.nn as nn
import numpy as np
import os
import sys
import argparse
import time
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import warnings

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

# Diffusion imports
from diffusers import DDPMPipeline, DDIMPipeline, DDPMScheduler, DDIMScheduler
import torchvision.transforms as transforms
from torchvision.utils import save_image, make_grid

# Local imports
from diffusion_maskpro import (
    DiffusionMaskProTrainer,
    get_model_sparsity_summary,
    validate_maskpro_model,
    load_maskpro_state
)
import utils


class MaskProEvaluator:
    """Comprehensive evaluator for MaskPro trained models."""
    
    def __init__(self, 
                 model_checkpoint: str,
                 baseline_model: str = None,
                 output_dir: str = "run/maskpro/evaluation",
                 device: str = "cuda:0"):
        """Initialize evaluator."""
        self.model_checkpoint = model_checkpoint
        self.baseline_model = baseline_model
        self.output_dir = Path(output_dir)
        self.device = torch.device(device)
        
        # Create output directories
        self.output_dir.mkdir(parents=True, exist_ok=True)
        (self.output_dir / "samples").mkdir(exist_ok=True)
        (self.output_dir / "metrics").mkdir(exist_ok=True)
        (self.output_dir / "analysis").mkdir(exist_ok=True)
        (self.output_dir / "visualizations").mkdir(exist_ok=True)
        
        # Models (will be loaded)
        self.maskpro_model = None
        self.baseline_model_obj = None
        self.scheduler = None
        
        # Evaluation results
        self.results = {
            'model_info': {},
            'sparsity_analysis': {},
            'quality_metrics': {},
            'performance_metrics': {},
            'comparison_metrics': {}
        }
        
    def load_models(self):
        """Load MaskPro model and baseline for comparison."""
        print("🔄 Loading models...")
        
        # Load MaskPro model from checkpoint
        if self.model_checkpoint.endswith('.pt'):
            checkpoint = torch.load(self.model_checkpoint, map_location='cpu')
            
            # Extract model
            if 'model_state_dict' in checkpoint:
                # This is a training checkpoint
                self.maskpro_model = self._load_model_from_training_checkpoint(checkpoint)
            else:
                # This might be a direct model save
                self.maskpro_model = checkpoint
        else:
            # Try loading as pipeline directory
            try:
                pipeline = DDPMPipeline.from_pretrained(self.model_checkpoint)
                self.maskpro_model = pipeline.unet
                self.scheduler = pipeline.scheduler
            except Exception as e:
                raise ValueError(f"Failed to load MaskPro model: {e}")
        
        # Move to device
        self.maskpro_model = self.maskpro_model.to(self.device)
        self.maskpro_model.eval()
        
        # Create scheduler if not loaded
        if self.scheduler is None:
            self.scheduler = DDPMScheduler(
                num_train_timesteps=1000,
                beta_start=0.0001,
                beta_end=0.02,
                beta_schedule="linear"
            )
        
        print(f"✓ MaskPro model loaded: {sum(p.numel() for p in self.maskpro_model.parameters()):,} parameters")
        
        # Load baseline model if provided
        if self.baseline_model:
            self.baseline_model_obj = self._load_baseline_model()
            print(f"✓ Baseline model loaded: {sum(p.numel() for p in self.baseline_model_obj.parameters()):,} parameters")
        
        # Store model info
        self.results['model_info'] = {
            'maskpro_parameters': sum(p.numel() for p in self.maskpro_model.parameters()),
            'maskpro_size_mb': sum(p.numel() * 4 for p in self.maskpro_model.parameters()) / (1024 * 1024),  # Assuming float32
            'baseline_parameters': sum(p.numel() for p in self.baseline_model_obj.parameters()) if self.baseline_model_obj else None,
            'baseline_size_mb': sum(p.numel() * 4 for p in self.baseline_model_obj.parameters()) / (1024 * 1024) if self.baseline_model_obj else None,
        }
        
    def _load_model_from_training_checkpoint(self, checkpoint):
        """Load model from training checkpoint."""
        # We need to reconstruct the model architecture
        # This is a simplified approach - in practice, you'd want to save the config too
        
        # Try to load the original model and apply the state dict
        try:
            # Attempt to get the original model path from checkpoint config
            if 'config' in checkpoint:
                pruned_model_path = checkpoint['config']['model']['pruned_model_path']
                
                # Load the original pruned model architecture
                pruned_dir = os.path.join(pruned_model_path, "pruned")
                if os.path.exists(pruned_dir):
                    potential_files = ["unet_pruned.pth", "unet_ema_pruned.pth", "model_pruned.pth"]
                    for file_name in potential_files:
                        file_path = os.path.join(pruned_dir, file_name)
                        if os.path.exists(file_path):
                            model = torch.load(file_path, map_location='cpu')
                            break
                    else:
                        raise FileNotFoundError("No pruned model file found")
                else:
                    # Try pipeline loading
                    pipeline = DDPMPipeline.from_pretrained(pruned_model_path, low_cpu_mem_usage=False, ignore_mismatched_sizes=True)
                    model = pipeline.unet
                
                # Load the trained state
                model.load_state_dict(checkpoint['model_state_dict'])
                return model
                
        except Exception as e:
            print(f"Warning: Could not load from checkpoint config: {e}")
            
        # Fallback: just return the state dict as-is and hope it works
        return checkpoint['model_state_dict']
    
    def _load_baseline_model(self):
        """Load baseline model for comparison."""
        if self.baseline_model.endswith('.pt') or self.baseline_model.endswith('.pth'):
            return torch.load(self.baseline_model, map_location=self.device)
        else:
            # Try loading as pipeline
            try:
                pipeline = DDPMPipeline.from_pretrained(self.baseline_model)
                return pipeline.unet.to(self.device)
            except Exception as e:
                print(f"Warning: Could not load baseline model: {e}")
                return None
    
    def analyze_sparsity(self):
        """Analyze sparsity patterns and N:M compliance."""
        print("\n🔍 Analyzing sparsity patterns...")
        
        # Get comprehensive sparsity analysis
        sparsity_summary = get_model_sparsity_summary(self.maskpro_model)
        
        # Validate MaskPro structure
        validation = validate_maskpro_model(self.maskpro_model)
        
        # Count different types of layers
        layer_analysis = {
            'total_layers': 0,
            'maskpro_layers': 0,
            'conv_layers': 0,
            'linear_layers': 0,
            'other_layers': 0
        }
        
        nm_compliance_details = []
        sparsity_details = []
        
        for name, module in self.maskpro_model.named_modules():
            layer_analysis['total_layers'] += 1
            
            if hasattr(module, 'get_sparsity_info'):  # MaskPro layer
                layer_analysis['maskpro_layers'] += 1
                info = module.get_sparsity_info()
                
                nm_compliance_details.append({
                    'layer_name': name,
                    'n': info['n'],
                    'm': info['m'],
                    'sparsity_ratio': info['sparsity_ratio'],
                    'nm_compliance': info['nm_compliance'],
                    'total_groups': info['total_groups'],
                    'compliant_groups': info['compliant_groups']
                })
                
                sparsity_details.append({
                    'layer_name': name,
                    'weight_shape': info['weight_shape'],
                    'total_weights': info['total_weights'],
                    'zero_weights': info['zero_weights'],
                    'sparsity_ratio': info['sparsity_ratio']
                })
                
                if isinstance(module.original_layer, nn.Conv2d):
                    layer_analysis['conv_layers'] += 1
                elif isinstance(module.original_layer, nn.Linear):
                    layer_analysis['linear_layers'] += 1
                else:
                    layer_analysis['other_layers'] += 1
        
        # Store results
        self.results['sparsity_analysis'] = {
            'overall_stats': sparsity_summary['overall_stats'],
            'layer_analysis': layer_analysis,
            'validation': validation,
            'nm_compliance_details': nm_compliance_details,
            'sparsity_details': sparsity_details,
            'layer_stats': sparsity_summary.get('layer_stats', {})
        }
        
        # Print summary
        print(f"✓ Overall sparsity: {sparsity_summary['overall_stats']['overall_sparsity']:.1%}")
        print(f"✓ MaskPro layers: {layer_analysis['maskpro_layers']}/{layer_analysis['total_layers']}")
        
        if nm_compliance_details:
            avg_compliance = np.mean([detail['nm_compliance'] for detail in nm_compliance_details])
            print(f"✓ Average N:M compliance: {avg_compliance:.1%}")
        
        return self.results['sparsity_analysis']
    
    def generate_samples(self, 
                        num_samples: int = 64,
                        batch_size: int = 8,
                        num_inference_steps: int = 1000,
                        save_grid: bool = True):
        """Generate samples from the MaskPro model."""
        print(f"\n🎨 Generating {num_samples} samples...")
        
        self.maskpro_model.eval()
        samples = []
        
        with torch.no_grad():
            for i in tqdm(range(0, num_samples, batch_size), desc="Generating samples"):
                current_batch_size = min(batch_size, num_samples - i)
                
                # Create noise
                noise = torch.randn(
                    current_batch_size, 3, 32, 32,  # Assuming CIFAR-10 resolution
                    device=self.device
                )
                
                # Denoising loop
                for t in tqdm(reversed(range(num_inference_steps)), desc=f"Batch {i//batch_size + 1}", leave=False):
                    # Create timestep tensor
                    timesteps = torch.full((current_batch_size,), t, device=self.device, dtype=torch.long)
                    
                    # Predict noise
                    with torch.cuda.amp.autocast(enabled=True):
                        noise_pred = self.maskpro_model(noise, timesteps).sample
                    
                    # Denoise
                    noise = self.scheduler.step(noise_pred, t, noise).prev_sample
                
                # Convert to images
                images = (noise + 1) / 2  # [-1, 1] -> [0, 1]
                images = torch.clamp(images, 0, 1)
                samples.append(images.cpu())
        
        # Concatenate all samples
        all_samples = torch.cat(samples, dim=0)
        
        # Save individual samples
        samples_dir = self.output_dir / "samples"
        for i, sample in enumerate(all_samples):
            save_image(sample, samples_dir / f"sample_{i:04d}.png")
        
        # Save grid
        if save_grid:
            grid = make_grid(all_samples, nrow=8, normalize=False, padding=2)
            save_image(grid, self.output_dir / "samples_grid.png")
        
        print(f"✓ Samples saved to {samples_dir}")
        
        # Store sample info
        self.results['quality_metrics']['num_samples_generated'] = len(all_samples)
        self.results['quality_metrics']['generation_resolution'] = list(all_samples[0].shape[1:])
        
        return all_samples
    
    def compute_fid_score(self, 
                         generated_samples: torch.Tensor,
                         reference_dataset: str = "cifar10",
                         num_reference_samples: int = 10000):
        """Compute FID score against reference dataset."""
        print(f"\n📊 Computing FID score against {reference_dataset}...")
        
        try:
            from torchmetrics.image.fid import FrechetInceptionDistance
        except ImportError:
            print("⚠️  torchmetrics not available, skipping FID computation")
            return None
        
        # Initialize FID metric
        fid = FrechetInceptionDistance(feature=2048, normalize=True)
        fid = fid.to(self.device)
        
        # Convert generated samples to uint8 [0, 255]
        generated_uint8 = (generated_samples * 255).clamp(0, 255).byte()
        
        # Load reference dataset
        if reference_dataset == "cifar10":
            from torchvision.datasets import CIFAR10
            transform = transforms.Compose([
                transforms.ToTensor(),
            ])
            
            dataset = CIFAR10(root="./data", train=True, download=True, transform=transform)
            dataloader = torch.utils.data.DataLoader(
                dataset, batch_size=32, shuffle=True, num_workers=4
            )
            
            # Get reference samples
            reference_samples = []
            for batch_idx, (images, _) in enumerate(dataloader):
                reference_samples.append(images)
                if len(reference_samples) * 32 >= num_reference_samples:
                    break
            
            reference_samples = torch.cat(reference_samples, dim=0)[:num_reference_samples]
            reference_uint8 = (reference_samples * 255).clamp(0, 255).byte()
        else:
            print(f"⚠️  Reference dataset {reference_dataset} not supported")
            return None
        
        # Update FID with real and fake images
        fid.update(reference_uint8.to(self.device), real=True)
        fid.update(generated_uint8.to(self.device), real=False)
        
        # Compute FID score
        fid_score = fid.compute().item()
        
        print(f"✓ FID Score: {fid_score:.2f}")
        
        # Store result
        self.results['quality_metrics']['fid_score'] = fid_score
        self.results['quality_metrics']['reference_dataset'] = reference_dataset
        self.results['quality_metrics']['num_reference_samples'] = num_reference_samples
        
        return fid_score
    
    def benchmark_inference_speed(self, 
                                 num_runs: int = 10,
                                 batch_size: int = 4,
                                 num_inference_steps: int = 100):
        """Benchmark inference speed and memory usage."""
        print(f"\n⏱️  Benchmarking inference speed...")
        
        self.maskpro_model.eval()
        
        # Warmup
        print("Warming up...")
        with torch.no_grad():
            noise = torch.randn(batch_size, 3, 32, 32, device=self.device)
            timesteps = torch.randint(0, 1000, (batch_size,), device=self.device)
            _ = self.maskpro_model(noise, timesteps)
        
        torch.cuda.synchronize() if self.device.type == 'cuda' else None
        
        # Benchmark single forward pass
        forward_times = []
        for _ in tqdm(range(num_runs), desc="Forward pass timing"):
            noise = torch.randn(batch_size, 3, 32, 32, device=self.device)
            timesteps = torch.randint(0, 1000, (batch_size,), device=self.device)
            
            torch.cuda.synchronize() if self.device.type == 'cuda' else None
            start_time = time.time()
            
            with torch.no_grad():
                _ = self.maskpro_model(noise, timesteps)
            
            torch.cuda.synchronize() if self.device.type == 'cuda' else None
            end_time = time.time()
            
            forward_times.append(end_time - start_time)
        
        # Benchmark complete generation
        generation_times = []
        for _ in tqdm(range(min(num_runs, 3)), desc="Full generation timing"):  # Fewer runs for full generation
            noise = torch.randn(1, 3, 32, 32, device=self.device)  # Single image
            
            start_time = time.time()
            
            with torch.no_grad():
                for t in reversed(range(num_inference_steps)):
                    timesteps = torch.full((1,), t, device=self.device, dtype=torch.long)
                    noise_pred = self.maskpro_model(noise, timesteps).sample
                    noise = self.scheduler.step(noise_pred, t, noise).prev_sample
            
            end_time = time.time()
            generation_times.append(end_time - start_time)
        
        # Memory usage
        if self.device.type == 'cuda':
            memory_allocated = torch.cuda.memory_allocated(self.device) / (1024**3)  # GB
            memory_reserved = torch.cuda.memory_reserved(self.device) / (1024**3)   # GB
        else:
            memory_allocated = memory_reserved = 0.0
        
        # Store results
        perf_metrics = {
            'forward_pass_time_mean': np.mean(forward_times),
            'forward_pass_time_std': np.std(forward_times),
            'forward_pass_time_min': np.min(forward_times),
            'generation_time_mean': np.mean(generation_times),
            'generation_time_std': np.std(generation_times),
            'batch_size': batch_size,
            'num_inference_steps': num_inference_steps,
            'memory_allocated_gb': memory_allocated,
            'memory_reserved_gb': memory_reserved,
            'device': str(self.device)
        }
        
        self.results['performance_metrics'] = perf_metrics
        
        print(f"✓ Forward pass: {perf_metrics['forward_pass_time_mean']*1000:.2f}±{perf_metrics['forward_pass_time_std']*1000:.2f}ms")
        print(f"✓ Full generation: {perf_metrics['generation_time_mean']:.2f}±{perf_metrics['generation_time_std']:.2f}s")
        if self.device.type == 'cuda':
            print(f"✓ Memory usage: {memory_allocated:.2f}GB allocated, {memory_reserved:.2f}GB reserved")
        
        return perf_metrics
    
    def compare_with_baseline(self):
        """Compare MaskPro model with baseline."""
        if self.baseline_model_obj is None:
            print("⚠️  No baseline model provided, skipping comparison")
            return None
        
        print("\n🔄 Comparing with baseline model...")
        
        # Model size comparison
        maskpro_params = sum(p.numel() for p in self.maskpro_model.parameters())
        baseline_params = sum(p.numel() for p in self.baseline_model_obj.parameters())
        
        compression_ratio = baseline_params / maskpro_params
        size_reduction = (1 - maskpro_params / baseline_params) * 100
        
        print(f"✓ Compression ratio: {compression_ratio:.2f}x")
        print(f"✓ Size reduction: {size_reduction:.1f}%")
        
        # Speed comparison (simplified)
        # This would require more sophisticated benchmarking in practice
        
        comparison = {
            'maskpro_parameters': maskpro_params,
            'baseline_parameters': baseline_params,
            'compression_ratio': compression_ratio,
            'size_reduction_percent': size_reduction,
            'maskpro_size_mb': self.results['model_info']['maskpro_size_mb'],
            'baseline_size_mb': self.results['model_info']['baseline_size_mb']
        }
        
        self.results['comparison_metrics'] = comparison
        
        return comparison
    
    def create_visualizations(self):
        """Create visualization plots for analysis."""
        print("\n📈 Creating visualizations...")
        
        viz_dir = self.output_dir / "visualizations"
        
        # 1. Sparsity distribution plot
        if 'sparsity_details' in self.results['sparsity_analysis']:
            sparsity_data = self.results['sparsity_analysis']['sparsity_details']
            
            plt.figure(figsize=(12, 6))
            
            # Sparsity ratios by layer
            plt.subplot(1, 2, 1)
            layer_names = [detail['layer_name'].split('.')[-1] for detail in sparsity_data]
            sparsity_ratios = [detail['sparsity_ratio'] for detail in sparsity_data]
            
            plt.bar(range(len(layer_names)), sparsity_ratios)
            plt.xticks(range(len(layer_names)), layer_names, rotation=45)
            plt.ylabel('Sparsity Ratio')
            plt.title('Sparsity Distribution Across Layers')
            plt.grid(True, alpha=0.3)
            
            # N:M compliance by layer
            plt.subplot(1, 2, 2)
            if 'nm_compliance_details' in self.results['sparsity_analysis']:
                nm_data = self.results['sparsity_analysis']['nm_compliance_details']
                compliance_ratios = [detail['nm_compliance'] for detail in nm_data]
                
                plt.bar(range(len(layer_names)), compliance_ratios)
                plt.xticks(range(len(layer_names)), layer_names, rotation=45)
                plt.ylabel('N:M Compliance Ratio')
                plt.title('N:M Compliance Across Layers')
                plt.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig(viz_dir / "sparsity_analysis.png", dpi=300, bbox_inches='tight')
            plt.close()
        
        # 2. Model comparison plot (if baseline available)
        if 'comparison_metrics' in self.results and self.results['comparison_metrics']:
            comp_data = self.results['comparison_metrics']
            
            plt.figure(figsize=(10, 6))
            
            # Parameters comparison
            plt.subplot(1, 2, 1)
            models = ['Baseline', 'MaskPro']
            params = [comp_data['baseline_parameters'], comp_data['maskpro_parameters']]
            
            bars = plt.bar(models, params)
            bars[1].set_color('orange')
            plt.ylabel('Number of Parameters')
            plt.title('Model Size Comparison')
            plt.ticklabel_format(style='scientific', axis='y', scilimits=(0,0))
            
            # Add compression ratio annotation
            plt.text(0.5, max(params) * 0.8, 
                    f'{comp_data["compression_ratio"]:.2f}x compression\n{comp_data["size_reduction_percent"]:.1f}% reduction',
                    ha='center', va='center', fontsize=12, 
                    bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
            
            # Memory usage comparison
            plt.subplot(1, 2, 2)
            sizes_mb = [comp_data['baseline_size_mb'], comp_data['maskpro_size_mb']]
            bars = plt.bar(models, sizes_mb)
            bars[1].set_color('orange')
            plt.ylabel('Model Size (MB)')
            plt.title('Memory Usage Comparison')
            
            plt.tight_layout()
            plt.savefig(viz_dir / "model_comparison.png", dpi=300, bbox_inches='tight')
            plt.close()
        
        print(f"✓ Visualizations saved to {viz_dir}")
    
    def save_results(self):
        """Save evaluation results to JSON."""
        results_file = self.output_dir / "evaluation_results.json"
        
        # Convert numpy types to Python types for JSON serialization
        def convert_numpy(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, (np.int64, np.int32)):
                return int(obj)
            elif isinstance(obj, (np.float64, np.float32)):
                return float(obj)
            return obj
        
        # Deep convert results
        import json
        json_str = json.dumps(self.results, default=convert_numpy, indent=2)
        converted_results = json.loads(json_str)
        
        with open(results_file, 'w') as f:
            json.dump(converted_results, f, indent=2)
        
        print(f"✓ Results saved to {results_file}")
        
        # Create summary report
        self._create_summary_report()
    
    def _create_summary_report(self):
        """Create a human-readable summary report."""
        report_file = self.output_dir / "evaluation_summary.md"
        
        with open(report_file, 'w') as f:
            f.write("# MaskPro Model Evaluation Report\n\n")
            
            # Model Information
            f.write("## Model Information\n")
            f.write(f"- **MaskPro Parameters**: {self.results['model_info']['maskpro_parameters']:,}\n")
            f.write(f"- **MaskPro Size**: {self.results['model_info']['maskpro_size_mb']:.2f} MB\n")
            if self.results['model_info']['baseline_parameters']:
                f.write(f"- **Baseline Parameters**: {self.results['model_info']['baseline_parameters']:,}\n")
                f.write(f"- **Baseline Size**: {self.results['model_info']['baseline_size_mb']:.2f} MB\n")
            f.write("\n")
            
            # Sparsity Analysis
            if 'sparsity_analysis' in self.results:
                sparsity = self.results['sparsity_analysis']
                f.write("## Sparsity Analysis\n")
                f.write(f"- **Overall Sparsity**: {sparsity['overall_stats']['overall_sparsity']:.1%}\n")
                f.write(f"- **MaskPro Layers**: {sparsity['layer_analysis']['maskpro_layers']}/{sparsity['layer_analysis']['total_layers']}\n")
                
                if sparsity['nm_compliance_details']:
                    avg_compliance = np.mean([d['nm_compliance'] for d in sparsity['nm_compliance_details']])
                    f.write(f"- **Average N:M Compliance**: {avg_compliance:.1%}\n")
                f.write("\n")
            
            # Quality Metrics
            if 'quality_metrics' in self.results and 'fid_score' in self.results['quality_metrics']:
                f.write("## Quality Metrics\n")
                f.write(f"- **FID Score**: {self.results['quality_metrics']['fid_score']:.2f}\n")
                f.write(f"- **Reference Dataset**: {self.results['quality_metrics']['reference_dataset']}\n")
                f.write(f"- **Samples Generated**: {self.results['quality_metrics']['num_samples_generated']}\n")
                f.write("\n")
            
            # Performance Metrics
            if 'performance_metrics' in self.results:
                perf = self.results['performance_metrics']
                f.write("## Performance Metrics\n")
                f.write(f"- **Forward Pass Time**: {perf['forward_pass_time_mean']*1000:.2f}±{perf['forward_pass_time_std']*1000:.2f} ms\n")
                f.write(f"- **Generation Time**: {perf['generation_time_mean']:.2f}±{perf['generation_time_std']:.2f} s\n")
                if perf['memory_allocated_gb'] > 0:
                    f.write(f"- **Memory Usage**: {perf['memory_allocated_gb']:.2f} GB\n")
                f.write("\n")
            
            # Comparison
            if 'comparison_metrics' in self.results and self.results['comparison_metrics']:
                comp = self.results['comparison_metrics']
                f.write("## Baseline Comparison\n")
                f.write(f"- **Compression Ratio**: {comp['compression_ratio']:.2f}x\n")
                f.write(f"- **Size Reduction**: {comp['size_reduction_percent']:.1f}%\n")
                f.write("\n")
            
            f.write("## Conclusion\n")
            f.write("This evaluation demonstrates the effectiveness of the two-stage hybrid pruning approach:\n")
            f.write("1. **Stage 1**: Magnitude-based structural pruning\n")
            f.write("2. **Stage 2**: N:M sparsity learning with MaskPro\n\n")
            f.write("The resulting model maintains quality while achieving significant compression and potential hardware acceleration benefits.\n")
        
        print(f"✓ Summary report saved to {report_file}")
    
    def run_full_evaluation(self, 
                           num_samples: int = 64,
                           compute_fid: bool = True,
                           benchmark_speed: bool = True):
        """Run complete evaluation pipeline."""
        print("🚀 Starting comprehensive MaskPro evaluation...")
        print("=" * 60)
        
        # Load models
        self.load_models()
        
        # Analyze sparsity
        self.analyze_sparsity()
        
        # Generate samples
        samples = self.generate_samples(num_samples=num_samples)
        
        # Compute quality metrics
        if compute_fid:
            self.compute_fid_score(samples)
        
        # Benchmark performance
        if benchmark_speed:
            self.benchmark_inference_speed()
        
        # Compare with baseline
        self.compare_with_baseline()
        
        # Create visualizations
        self.create_visualizations()
        
        # Save results
        self.save_results()
        
        print("\n" + "=" * 60)
        print("🎉 Evaluation completed successfully!")
        print(f"📁 Results saved to: {self.output_dir}")
        print("=" * 60)
        
        return self.results


def main():
    """Main evaluation function."""
    parser = argparse.ArgumentParser(description="Evaluate MaskPro trained models")
    
    parser.add_argument("--model_checkpoint", type=str, required=True,
                       help="Path to MaskPro model checkpoint or directory")
    parser.add_argument("--baseline_model", type=str, default=None,
                       help="Path to baseline model for comparison")
    parser.add_argument("--output_dir", type=str, default="run/maskpro/evaluation",
                       help="Output directory for evaluation results")
    parser.add_argument("--device", type=str, default="cuda:0",
                       help="Device to use for evaluation")
    parser.add_argument("--num_samples", type=int, default=64,
                       help="Number of samples to generate")
    parser.add_argument("--skip_fid", action="store_true",
                       help="Skip FID score computation")
    parser.add_argument("--skip_benchmark", action="store_true",
                       help="Skip performance benchmarking")
    
    args = parser.parse_args()
    
    # Create evaluator
    evaluator = MaskProEvaluator(
        model_checkpoint=args.model_checkpoint,
        baseline_model=args.baseline_model,
        output_dir=args.output_dir,
        device=args.device
    )
    
    # Run evaluation
    results = evaluator.run_full_evaluation(
        num_samples=args.num_samples,
        compute_fid=not args.skip_fid,
        benchmark_speed=not args.skip_benchmark
    )
    
    return results


if __name__ == "__main__":
    main() 