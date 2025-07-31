#!/usr/bin/env python3
"""
MaskPro Evaluation Workflow

This script provides a comprehensive evaluation workflow for MaskPro models:
1. Quick evaluation during training
2. Full post-training evaluation  
3. Multi-model comparison
4. Analysis and reporting

Usage Examples:
    # Quick evaluation of latest checkpoint
    python evaluation_workflow.py --mode quick --checkpoint run/maskpro/training/checkpoints/best_checkpoint.pt
    
    # Full evaluation with FID and benchmarks
    python evaluation_workflow.py --mode full --checkpoint run/maskpro/training/checkpoints/best_checkpoint.pt
    
    # Compare multiple models
    python evaluation_workflow.py --mode compare --maskpro run/maskpro/training/checkpoints/best_checkpoint.pt --magnitude run/pruned/magnitude/ddpm_cifar10_pruned
"""

import os
import sys
import argparse
import subprocess
from pathlib import Path
from typing import Dict, List, Optional
import json
import time

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

class EvaluationWorkflow:
    """Orchestrate different evaluation modes."""
    
    def __init__(self, output_base_dir: str = "run/maskpro/evaluation"):
        self.output_base_dir = Path(output_base_dir)
        self.output_base_dir.mkdir(parents=True, exist_ok=True)
        
        # Evaluation script paths
        self.scripts_dir = Path(__file__).parent
        self.quick_eval_script = self.scripts_dir / "quick_evaluate.py"
        self.full_eval_script = self.scripts_dir / "evaluate_maskpro_model.py"
        self.compare_script = self.scripts_dir / "compare_models.py"
        
        self.results = {}
    
    def run_quick_evaluation(self, checkpoint: str, device: str = "cuda:0") -> Dict:
        """Run quick evaluation for training monitoring."""
        print("🚀 Running Quick Evaluation")
        print("=" * 50)
        
        output_dir = self.output_base_dir / "quick"
        
        cmd = [
            sys.executable, str(self.quick_eval_script),
            "--checkpoint", checkpoint,
            "--device", device,
            "--output_dir", str(output_dir),
            "--num_samples", "8"
        ]
        
        print(f"Command: {' '.join(cmd)}")
        
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            print("✓ Quick evaluation completed successfully")
            print(result.stdout)
            
            return {
                'status': 'success',
                'output_dir': str(output_dir),
                'stdout': result.stdout,
                'stderr': result.stderr
            }
            
        except subprocess.CalledProcessError as e:
            print(f"❌ Quick evaluation failed: {e}")
            print(f"STDOUT: {e.stdout}")
            print(f"STDERR: {e.stderr}")
            
            return {
                'status': 'failed',
                'error': str(e),
                'stdout': e.stdout,
                'stderr': e.stderr
            }
    
    def run_full_evaluation(self, 
                           checkpoint: str, 
                           baseline: Optional[str] = None,
                           device: str = "cuda:0",
                           num_samples: int = 64,
                           skip_fid: bool = False,
                           skip_benchmark: bool = False) -> Dict:
        """Run comprehensive evaluation."""
        print("🚀 Running Full Evaluation")
        print("=" * 50)
        
        output_dir = self.output_base_dir / "full"
        
        cmd = [
            sys.executable, str(self.full_eval_script),
            "--model_checkpoint", checkpoint,
            "--device", device,
            "--output_dir", str(output_dir),
            "--num_samples", str(num_samples)
        ]
        
        if baseline:
            cmd.extend(["--baseline_model", baseline])
        
        if skip_fid:
            cmd.append("--skip_fid")
            
        if skip_benchmark:
            cmd.append("--skip_benchmark")
        
        print(f"Command: {' '.join(cmd)}")
        
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            print("✓ Full evaluation completed successfully")
            
            # Try to load results
            results_file = output_dir / "evaluation_results.json"
            if results_file.exists():
                with open(results_file, 'r') as f:
                    evaluation_results = json.load(f)
            else:
                evaluation_results = {}
            
            return {
                'status': 'success',
                'output_dir': str(output_dir),
                'results': evaluation_results,
                'stdout': result.stdout,
                'stderr': result.stderr
            }
            
        except subprocess.CalledProcessError as e:
            print(f"❌ Full evaluation failed: {e}")
            print(f"STDERR: {e.stderr}")
            
            return {
                'status': 'failed',
                'error': str(e),
                'stdout': e.stdout,
                'stderr': e.stderr
            }
    
    def run_model_comparison(self,
                            maskpro_checkpoint: Optional[str] = None,
                            magnitude_model: Optional[str] = None,
                            original_model: Optional[str] = None,
                            additional_models: List[str] = None,
                            device: str = "cuda:0") -> Dict:
        """Run multi-model comparison."""
        print("🚀 Running Model Comparison")
        print("=" * 50)
        
        output_dir = self.output_base_dir / "comparison"
        
        cmd = [
            sys.executable, str(self.compare_script),
            "--device", device,
            "--output_dir", str(output_dir)
        ]
        
        if original_model:
            cmd.extend(["--original", original_model])
            
        if magnitude_model:
            cmd.extend(["--magnitude", magnitude_model])
            
        if maskpro_checkpoint:
            cmd.extend(["--maskpro", maskpro_checkpoint])
        
        if additional_models:
            cmd.extend(["--additional"] + additional_models)
        
        print(f"Command: {' '.join(cmd)}")
        
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            print("✓ Model comparison completed successfully")
            
            return {
                'status': 'success',
                'output_dir': str(output_dir),
                'stdout': result.stdout,
                'stderr': result.stderr
            }
            
        except subprocess.CalledProcessError as e:
            print(f"❌ Model comparison failed: {e}")
            print(f"STDERR: {e.stderr}")
            
            return {
                'status': 'failed',
                'error': str(e),
                'stdout': e.stdout,
                'stderr': e.stderr
            }
    
    def run_comprehensive_workflow(self,
                                  checkpoint: str,
                                  magnitude_model: Optional[str] = None,
                                  original_model: Optional[str] = None,
                                  device: str = "cuda:0") -> Dict:
        """Run complete evaluation workflow."""
        print("🎯 Starting Comprehensive Evaluation Workflow")
        print("=" * 60)
        
        workflow_results = {
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'checkpoint': checkpoint,
            'device': device,
            'steps': {}
        }
        
        # Step 1: Quick evaluation
        print("\n📋 Step 1: Quick Evaluation")
        quick_results = self.run_quick_evaluation(checkpoint, device)
        workflow_results['steps']['quick_evaluation'] = quick_results
        
        if quick_results['status'] != 'success':
            print("⚠️  Quick evaluation failed, continuing with other steps...")
        
        # Step 2: Full evaluation
        print("\n📋 Step 2: Full Evaluation")
        full_results = self.run_full_evaluation(
            checkpoint=checkpoint,
            baseline=magnitude_model,
            device=device,
            num_samples=64
        )
        workflow_results['steps']['full_evaluation'] = full_results
        
        if full_results['status'] != 'success':
            print("⚠️  Full evaluation failed, continuing with comparison...")
        
        # Step 3: Model comparison (if other models available)
        if magnitude_model or original_model:
            print("\n📋 Step 3: Model Comparison")
            comparison_results = self.run_model_comparison(
                maskpro_checkpoint=checkpoint,
                magnitude_model=magnitude_model,
                original_model=original_model,
                device=device
            )
            workflow_results['steps']['model_comparison'] = comparison_results
        else:
            print("\n📋 Step 3: Model Comparison (Skipped - no baseline models)")
            workflow_results['steps']['model_comparison'] = {'status': 'skipped', 'reason': 'no_baseline_models'}
        
        # Save workflow results
        self.save_workflow_results(workflow_results)
        
        # Print summary
        self.print_workflow_summary(workflow_results)
        
        return workflow_results
    
    def save_workflow_results(self, results: Dict):
        """Save workflow results to file."""
        results_file = self.output_base_dir / "workflow_results.json"
        
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"\n💾 Workflow results saved to: {results_file}")
    
    def print_workflow_summary(self, results: Dict):
        """Print workflow summary."""
        print("\n" + "=" * 60)
        print("📊 EVALUATION WORKFLOW SUMMARY")
        print("=" * 60)
        
        print(f"🕐 Timestamp: {results['timestamp']}")
        print(f"📂 Checkpoint: {results['checkpoint']}")
        print(f"💻 Device: {results['device']}")
        
        print("\n📋 Steps Summary:")
        
        for step_name, step_results in results['steps'].items():
            status_emoji = "✅" if step_results['status'] == 'success' else "❌" if step_results['status'] == 'failed' else "⏭️"
            print(f"  {status_emoji} {step_name.replace('_', ' ').title()}: {step_results['status'].upper()}")
            
            if step_results['status'] == 'success' and 'output_dir' in step_results:
                print(f"     📁 Output: {step_results['output_dir']}")
        
        # Quick stats if available
        if 'full_evaluation' in results['steps'] and results['steps']['full_evaluation']['status'] == 'success':
            full_results = results['steps']['full_evaluation'].get('results', {})
            
            if 'sparsity_analysis' in full_results:
                sparsity = full_results['sparsity_analysis']['overall_stats']['overall_sparsity']
                print(f"\n🎯 Key Metrics:")
                print(f"   Sparsity: {sparsity:.1%}")
                
            if 'quality_metrics' in full_results and 'fid_score' in full_results['quality_metrics']:
                fid = full_results['quality_metrics']['fid_score']
                print(f"   FID Score: {fid:.2f}")
                
            if 'performance_metrics' in full_results:
                perf = full_results['performance_metrics']
                print(f"   Inference: {perf['forward_pass_time_mean']*1000:.2f}ms")
        
        print("\n📁 All results saved to:", self.output_base_dir)
        
        # Next steps recommendation
        print("\n🔄 Recommended Next Steps:")
        print("1. Review generated samples in samples/ directories")
        print("2. Check evaluation reports (.md files)")
        print("3. Analyze comparison visualizations (.png files)")
        print("4. Use results for model selection and optimization")
        
        print("=" * 60)


def find_latest_checkpoint(training_dir: str) -> Optional[str]:
    """Find the latest checkpoint in training directory."""
    training_path = Path(training_dir)
    checkpoints_dir = training_path / "checkpoints"
    
    if not checkpoints_dir.exists():
        return None
    
    # Look for best checkpoint first
    best_checkpoint = checkpoints_dir / "best_checkpoint.pt"
    if best_checkpoint.exists():
        return str(best_checkpoint)
    
    # Look for latest epoch checkpoint
    checkpoint_files = list(checkpoints_dir.glob("checkpoint_epoch_*.pt"))
    if checkpoint_files:
        # Sort by epoch number
        def extract_epoch(path):
            try:
                return int(path.stem.split('_')[-1])
            except:
                return -1
        
        latest = max(checkpoint_files, key=extract_epoch)
        return str(latest)
    
    return None


def main():
    """Main evaluation workflow function."""
    parser = argparse.ArgumentParser(description="MaskPro Evaluation Workflow")
    
    # Mode selection
    parser.add_argument("--mode", type=str, choices=['quick', 'full', 'compare', 'comprehensive'], 
                       default='comprehensive',
                       help="Evaluation mode to run")
    
    # Model paths
    parser.add_argument("--checkpoint", type=str, default=None,
                       help="Path to MaskPro checkpoint (if None, searches latest)")
    parser.add_argument("--training_dir", type=str, default="run/maskpro/training",
                       help="Training directory to search for checkpoints")
    parser.add_argument("--magnitude_model", type=str, default="run/pruned/magnitude/ddpm_cifar10_pruned",
                       help="Path to magnitude-pruned model")
    parser.add_argument("--original_model", type=str, default=None,
                       help="Path to original unpruned model")
    
    # Evaluation options
    parser.add_argument("--device", type=str, default="cuda:0",
                       help="Device to use for evaluation")
    parser.add_argument("--output_dir", type=str, default="run/maskpro/evaluation",
                       help="Base output directory")
    parser.add_argument("--num_samples", type=int, default=64,
                       help="Number of samples for full evaluation")
    parser.add_argument("--skip_fid", action="store_true",
                       help="Skip FID score computation")
    parser.add_argument("--skip_benchmark", action="store_true",
                       help="Skip performance benchmarking")
    
    # Additional models for comparison
    parser.add_argument("--additional_models", type=str, nargs='*', default=[],
                       help="Additional models for comparison (format: name:path:description)")
    
    args = parser.parse_args()
    
    # Find checkpoint if not provided
    checkpoint = args.checkpoint
    if checkpoint is None:
        print(f"🔍 Searching for latest checkpoint in {args.training_dir}...")
        checkpoint = find_latest_checkpoint(args.training_dir)
        
        if checkpoint is None:
            print(f"❌ No checkpoint found in {args.training_dir}")
            print("Please specify --checkpoint or ensure training has produced checkpoints")
            return 1
        else:
            print(f"✓ Found checkpoint: {checkpoint}")
    
    # Verify checkpoint exists
    if not os.path.exists(checkpoint):
        print(f"❌ Checkpoint not found: {checkpoint}")
        return 1
    
    # Create workflow
    workflow = EvaluationWorkflow(args.output_dir)
    
    # Run evaluation based on mode
    if args.mode == 'quick':
        results = workflow.run_quick_evaluation(checkpoint, args.device)
        
    elif args.mode == 'full':
        baseline = args.magnitude_model if os.path.exists(args.magnitude_model) else None
        results = workflow.run_full_evaluation(
            checkpoint=checkpoint,
            baseline=baseline,
            device=args.device,
            num_samples=args.num_samples,
            skip_fid=args.skip_fid,
            skip_benchmark=args.skip_benchmark
        )
        
    elif args.mode == 'compare':
        magnitude_model = args.magnitude_model if os.path.exists(args.magnitude_model) else None
        original_model = args.original_model if args.original_model and os.path.exists(args.original_model) else None
        
        results = workflow.run_model_comparison(
            maskpro_checkpoint=checkpoint,
            magnitude_model=magnitude_model,
            original_model=original_model,
            additional_models=args.additional_models,
            device=args.device
        )
        
    elif args.mode == 'comprehensive':
        magnitude_model = args.magnitude_model if os.path.exists(args.magnitude_model) else None
        original_model = args.original_model if args.original_model and os.path.exists(args.original_model) else None
        
        results = workflow.run_comprehensive_workflow(
            checkpoint=checkpoint,
            magnitude_model=magnitude_model,
            original_model=original_model,
            device=args.device
        )
    
    # Return appropriate exit code
    if isinstance(results, dict):
        if results.get('status') == 'success':
            return 0
        elif 'steps' in results:
            # Comprehensive mode - check if any step succeeded
            success_count = sum(1 for step in results['steps'].values() if step.get('status') == 'success')
            return 0 if success_count > 0 else 1
        else:
            return 1
    else:
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code) 