"""
Diffusion MaskPro Trainer

This module implements a comprehensive training framework for learning N:M sparse masks
on magnitude-pruned diffusion models using policy gradient optimization.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import os
import time
import yaml
import math
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from tqdm import tqdm
import warnings

# Diffusion model imports
from diffusers import DDPMPipeline, DDIMPipeline, DDPMScheduler, DDIMScheduler
from diffusers.optimization import get_cosine_schedule_with_warmup

# SwanLab for logging
try:
    import swanlab
    SWANLAB_AVAILABLE = True
except ImportError:
    SWANLAB_AVAILABLE = False
    warnings.warn("SwanLab not available. Logging will be limited.")

# Local imports
from .maskpro_layer import MaskProLayer
from .utils import (
    wrap_model_with_maskpro, 
    get_model_sparsity_summary,
    validate_maskpro_model,
    save_maskpro_state
)
import utils  # Project utils for dataset loading


class DiffusionMaskProTrainer:
    """
    Comprehensive trainer for learning N:M sparse masks on diffusion models.
    
    This trainer implements the second stage of the hybrid pruning approach:
    1. Load magnitude-pruned model (Stage 1 output)
    2. Apply MaskPro wrappers for N:M sparsity learning
    3. Train using policy gradient optimization
    4. Monitor progress with SwanLab integration
    """
    
    def __init__(self, config_path: str):
        """Initialize trainer with configuration."""
        self.config = self._load_config(config_path)
        self.device = torch.device(self.config['hardware']['device'])
        
        # Training state
        self.current_epoch = 0
        self.current_step = 0
        self.best_val_loss = float('inf')
        self.train_start_time = None
        
        # Model components (will be initialized)
        self.model = None
        self.scheduler = None
        self.pipeline = None
        self.optimizers = {}
        self.lr_schedulers = {}
        
        # Data loaders
        self.train_loader = None
        self.val_loader = None
        
        # Logging
        self.swanlab_run = None
        self.training_logs = []
        
        # Initialize components
        self._setup_logging()
        self._setup_directories()
        
    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """Load and validate configuration."""
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        # Validate required keys
        required_keys = ['model', 'training', 'dataset', 'hardware', 'output']
        for key in required_keys:
            if key not in config:
                raise ValueError(f"Missing required configuration key: {key}")
        
        return config
    
    def _setup_logging(self):
        """Initialize SwanLab logging."""
        if not self.config['logging'].get('use_swanlab', False):
            print("SwanLab logging disabled")
            return
            
        if not SWANLAB_AVAILABLE:
            print("SwanLab not available, skipping logging setup")
            return
        
        # Initialize SwanLab run
        self.swanlab_run = swanlab.init(
            project=self.config['logging']['project_name'],
            experiment_name=self.config['logging']['experiment_name'],
            config=self.config,
            logdir=os.path.join(self.config['output']['output_dir'], 
                               self.config['output']['logs_dir'])
        )
        
        print(f"✓ SwanLab logging initialized: {self.config['logging']['experiment_name']}")
    
    def _setup_directories(self):
        """Create necessary output directories."""
        base_dir = self.config['output']['output_dir']
        subdirs = [
            self.config['output']['checkpoints_dir'],
            self.config['output']['logs_dir'],
            self.config['output']['samples_dir'],
            self.config['output']['analysis_dir']
        ]
        
        for subdir in subdirs:
            full_path = os.path.join(base_dir, subdir)
            os.makedirs(full_path, exist_ok=True)
        
        print(f"✓ Output directories created in: {base_dir}")
    
    def load_model_and_scheduler(self):
        """Load the magnitude-pruned model and initialize scheduler."""
        print("Loading magnitude-pruned model...")
        
        model_path = self.config['model']['pruned_model_path']
        
        # Load model using same strategy as mask extraction
        pruned_dir = os.path.join(model_path, "pruned")
        
        if os.path.exists(pruned_dir):
            # Load direct pruned model
            potential_files = ["unet_pruned.pth", "unet_ema_pruned.pth", "model_pruned.pth"]
            
            model_file = None
            for file_name in potential_files:
                file_path = os.path.join(pruned_dir, file_name)
                if os.path.exists(file_path):
                    model_file = file_path
                    break
            
            if model_file is None:
                raise FileNotFoundError(f"No pruned model file found in {pruned_dir}")
            
            print(f"Loading model from: {model_file}")
            self.model = torch.load(model_file, map_location='cpu')
            
            # Create scheduler
            self.scheduler = DDPMScheduler(
                num_train_timesteps=self.config['training']['num_train_timesteps'],
                beta_start=0.0001,
                beta_end=0.02,
                beta_schedule="linear"
            )
            
        else:
            # Try pipeline loading as fallback
            try:
                if "ddpm" in model_path.lower():
                    pipeline = DDPMPipeline.from_pretrained(
                        model_path,
                        low_cpu_mem_usage=False,
                        ignore_mismatched_sizes=True
                    )
                    self.scheduler = DDPMScheduler.from_pretrained(model_path, subfolder="scheduler")
                else:
                    pipeline = DDIMPipeline.from_pretrained(
                        model_path,
                        low_cpu_mem_usage=False,
                        ignore_mismatched_sizes=True
                    )
                    self.scheduler = DDIMScheduler.from_pretrained(model_path, subfolder="scheduler")
                
                self.model = pipeline.unet
                print("✓ Loaded as pipeline with size mismatch handling")
                
            except Exception as e:
                raise ValueError(f"Failed to load model: {e}")
        
        # Move to device
        self.model = self.model.to(self.device)
        self.model.train()
        
        # Print model info
        total_params = sum(p.numel() for p in self.model.parameters())
        print(f"✓ Model loaded: {total_params:,} parameters")
        
        return self.model, self.scheduler
    
    def apply_maskpro_wrappers(self):
        """Apply MaskPro wrappers to the model."""
        print("Applying MaskPro wrappers...")
        
        model_config = self.config['model']
        
        # Wrap model with MaskPro layers
        wrap_log = wrap_model_with_maskpro(
            self.model,
            n=model_config['n'],
            m=model_config['m'],
            target_layers=model_config['target_layers'],
            exclude_layers=model_config['exclude_layers']
        )
        
        wrapped_count = sum(1 for status in wrap_log.values() if "✓" in status)
        print(f"✓ Wrapped {wrapped_count} layers with MaskPro")
        
        # Load initial masks if available
        initial_masks_dir = model_config.get('initial_masks_dir')
        if initial_masks_dir and os.path.exists(initial_masks_dir):
            self._load_initial_masks(initial_masks_dir)
        
        # Validate wrapped model
        validation = validate_maskpro_model(self.model)
        if not validation['is_valid']:
            raise RuntimeError(f"Model validation failed: {validation['errors']}")
        
        print(f"✓ Model validation passed: {validation['layer_count']} MaskPro layers")
        
        # Log initial sparsity
        sparsity_summary = get_model_sparsity_summary(self.model)
        print(f"✓ Initial sparsity: {sparsity_summary['overall_stats']['overall_sparsity']:.1%}")
        
        return wrap_log
    
    def _load_initial_masks(self, masks_dir: str):
        """Load initial masks from extracted files."""
        print(f"Loading initial masks from: {masks_dir}")
        
        mask_files = [f for f in os.listdir(masks_dir) if f.endswith('.pt')]
        if not mask_files:
            print("Warning: No initial mask files found")
            return
        
        masks_loaded = 0
        for name, module in self.model.named_modules():
            if isinstance(module, MaskProLayer):
                # Try to find corresponding mask file
                for mask_file in mask_files:
                    layer_name = mask_file.replace('.pt', '').replace('_', '.')
                    if layer_name in name or name in layer_name:
                        try:
                            mask_path = os.path.join(masks_dir, mask_file)
                            mask = torch.load(mask_path, map_location='cpu')
                            module.current_mask.copy_(mask.to(self.device))
                            masks_loaded += 1
                            break
                        except Exception as e:
                            print(f"Warning: Failed to load mask for {name}: {e}")
        
        print(f"✓ Loaded initial masks for {masks_loaded} layers")
    
    def setup_optimizers(self):
        """Setup dual optimizers for model and mask parameters."""
        print("Setting up optimizers...")
        
        # Separate parameters
        mask_params = [p for name, p in self.model.named_parameters() if 'mask_logits' in name]
        model_params = [p for name, p in self.model.named_parameters() if 'mask_logits' not in name]
        
        training_config = self.config['training']
        
        # Model optimizer
        self.optimizers['model'] = optim.AdamW(
            model_params,
            lr=training_config['model_lr'],
            weight_decay=training_config['weight_decay']
        )
        
        # Mask optimizer  
        self.optimizers['mask'] = optim.AdamW(
            mask_params,
            lr=training_config['mask_lr'],
            weight_decay=training_config['weight_decay'] * 0.1  # Lower weight decay for masks
        )
        
        # Learning rate schedulers
        total_steps = len(self.train_loader) * training_config['epochs']
        
        self.lr_schedulers['model'] = get_cosine_schedule_with_warmup(
            self.optimizers['model'],
            num_warmup_steps=total_steps * 0.1,
            num_training_steps=total_steps
        )
        
        self.lr_schedulers['mask'] = get_cosine_schedule_with_warmup(
            self.optimizers['mask'],
            num_warmup_steps=total_steps * 0.05,  # Shorter warmup for masks
            num_training_steps=total_steps
        )
        
        print(f"✓ Optimizers setup: {len(model_params)} model params, {len(mask_params)} mask params")
        
        return self.optimizers, self.lr_schedulers
    
    def setup_data_loaders(self):
        """Setup training and validation data loaders."""
        print("Setting up data loaders...")
        
        dataset_config = self.config['dataset']
        
        # Get dataset
        dataset_name = dataset_config['name']
        if dataset_name == "cifar10":
            full_dataset = utils.get_dataset("cifar10")
        else:
            raise ValueError(f"Unsupported dataset: {dataset_name}")
        
        # Split dataset
        dataset_size = dataset_config['size']
        val_size = dataset_config['val_size']
        
        if len(full_dataset) < dataset_size + val_size:
            print(f"Warning: Dataset too small, using all {len(full_dataset)} samples")
            dataset_size = int(len(full_dataset) * 0.8)
            val_size = len(full_dataset) - dataset_size
        
        # Create random split
        indices = torch.randperm(len(full_dataset))
        train_indices = indices[:dataset_size]
        val_indices = indices[dataset_size:dataset_size + val_size]
        
        train_dataset = torch.utils.data.Subset(full_dataset, train_indices)
        val_dataset = torch.utils.data.Subset(full_dataset, val_indices)
        
        # Create data loaders
        self.train_loader = DataLoader(
            train_dataset,
            batch_size=self.config['training']['batch_size'],
            shuffle=True,
            num_workers=dataset_config['num_workers'],
            pin_memory=dataset_config['pin_memory'],
            drop_last=True
        )
        
        self.val_loader = DataLoader(
            val_dataset,
            batch_size=self.config['training']['batch_size'],
            shuffle=False,
            num_workers=dataset_config['num_workers'],
            pin_memory=dataset_config['pin_memory'],
            drop_last=False
        )
        
        print(f"✓ Data loaders created: {len(train_dataset)} train, {len(val_dataset)} val")
        
        return self.train_loader, self.val_loader
    
    def compute_diffusion_loss(self, batch: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Compute diffusion denoising loss."""
        batch = batch.to(self.device)
        batch_size = batch.size(0)
        
        # Sample noise and timesteps
        noise = torch.randn_like(batch)
        timesteps = torch.randint(
            0, 
            self.scheduler.config.num_train_timesteps, 
            (batch_size,), 
            device=self.device
        )
        
        # Add noise to images
        noisy_images = self.scheduler.add_noise(batch, noise, timesteps)
        
        # Predict noise
        if self.config['hardware']['mixed_precision']:
            with torch.cuda.amp.autocast():
                noise_pred = self.model(noisy_images, timesteps).sample
                main_loss = F.mse_loss(noise_pred, noise, reduction='mean')
        else:
            noise_pred = self.model(noisy_images, timesteps).sample
            main_loss = F.mse_loss(noise_pred, noise, reduction='mean')
        
        # Additional metrics
        metrics = {
            'main_loss': main_loss.item(),
            'mse_loss': F.mse_loss(noise_pred, noise, reduction='mean').item(),
            'noise_norm': noise.norm().item(),
            'pred_norm': noise_pred.norm().item()
        }
        
        return main_loss, metrics
    
    def compute_mask_loss(self, main_loss: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Compute policy gradient loss for mask learning."""
        total_mask_loss = torch.tensor(0.0, device=self.device)
        mask_count = 0
        
        for module in self.model.modules():
            if isinstance(module, MaskProLayer):
                mask_loss = module.get_mask_loss(main_loss)
                total_mask_loss += mask_loss
                mask_count += 1
        
        # Normalize by number of masked layers
        if mask_count > 0:
            total_mask_loss = total_mask_loss / mask_count
        
        # Apply mask loss weight
        weighted_mask_loss = total_mask_loss * self.config['training']['mask_loss_weight']
        
        metrics = {
            'mask_loss': total_mask_loss.item(),
            'weighted_mask_loss': weighted_mask_loss.item(),
            'mask_layers': mask_count
        }
        
        return weighted_mask_loss, metrics
    
    def training_step(self, batch: torch.Tensor) -> Dict[str, float]:
        """Execute single training step."""
        # Handle different batch formats
        if isinstance(batch, (list, tuple)):
            batch = batch[0]  # Take images only
        
        step_metrics = {}
        
        # Forward pass and main loss
        main_loss, main_metrics = self.compute_diffusion_loss(batch)
        step_metrics.update(main_metrics)
        
        # Update model parameters
        self.optimizers['model'].zero_grad()
        
        if self.config['hardware']['mixed_precision']:
            # Mixed precision training
            scaler = torch.cuda.amp.GradScaler()
            scaler.scale(main_loss).backward(retain_graph=True)
            
            # Gradient clipping
            if self.config['training']['gradient_clip_norm'] > 0:
                scaler.unscale_(self.optimizers['model'])
                torch.nn.utils.clip_grad_norm_(
                    [p for name, p in self.model.named_parameters() if 'mask_logits' not in name],
                    self.config['training']['gradient_clip_norm']
                )
            
            scaler.step(self.optimizers['model'])
            scaler.update()
        else:
            main_loss.backward(retain_graph=True)
            
            # Gradient clipping
            if self.config['training']['gradient_clip_norm'] > 0:
                torch.nn.utils.clip_grad_norm_(
                    [p for name, p in self.model.named_parameters() if 'mask_logits' not in name],
                    self.config['training']['gradient_clip_norm']
                )
            
            self.optimizers['model'].step()
        
        # Update learning rate
        self.lr_schedulers['model'].step()
        
        # Compute and update mask loss
        mask_loss, mask_metrics = self.compute_mask_loss(main_loss.detach())
        step_metrics.update(mask_metrics)
        
        if mask_loss.requires_grad:
            self.optimizers['mask'].zero_grad()
            mask_loss.backward()
            
            # Gradient clipping for mask parameters
            if self.config['training']['gradient_clip_norm'] > 0:
                torch.nn.utils.clip_grad_norm_(
                    [p for name, p in self.model.named_parameters() if 'mask_logits' in name],
                    self.config['training']['gradient_clip_norm']
                )
            
            self.optimizers['mask'].step()
            self.lr_schedulers['mask'].step()
        
        # Total loss for logging
        total_loss = main_loss + mask_loss
        step_metrics['total_loss'] = total_loss.item()
        
        # Learning rates
        step_metrics['model_lr'] = self.optimizers['model'].param_groups[0]['lr']
        step_metrics['mask_lr'] = self.optimizers['mask'].param_groups[0]['lr']
        
        # Sparsity metrics
        sparsity_summary = get_model_sparsity_summary(self.model)
        step_metrics['sparsity_ratio'] = sparsity_summary['overall_stats']['overall_sparsity']
        
        # N:M compliance (sample a few layers)
        nm_compliance_sum = 0
        nm_count = 0
        for module in self.model.modules():
            if isinstance(module, MaskProLayer):
                info = module.get_sparsity_info()
                nm_compliance_sum += info['nm_compliance']
                nm_count += 1
                if nm_count >= 5:  # Sample only first 5 layers for efficiency
                    break
        
        if nm_count > 0:
            step_metrics['nm_compliance'] = nm_compliance_sum / nm_count
        
        return step_metrics
    
    def log_metrics(self, metrics: Dict[str, float], step: int, prefix: str = "train"):
        """Log metrics to SwanLab and console."""
        # Add prefix to metrics
        prefixed_metrics = {f"{prefix}/{key}": value for key, value in metrics.items()}
        
        # Log to SwanLab
        if self.swanlab_run is not None:
            self.swanlab_run.log(prefixed_metrics, step=step)
        
        # Store in training logs
        log_entry = {
            'step': step,
            'epoch': self.current_epoch,
            'prefix': prefix,
            **metrics
        }
        self.training_logs.append(log_entry)
        
        # Console logging (less frequent)
        if step % self.config['logging']['log_freq'] == 0:
            metrics_str = " | ".join([f"{k}: {v:.6f}" for k, v in metrics.items() if k in [
                'main_loss', 'mask_loss', 'total_loss', 'sparsity_ratio', 'nm_compliance'
            ]])
            print(f"[{prefix.upper()}] Epoch {self.current_epoch}, Step {step}: {metrics_str}")
    
    def validate(self) -> Dict[str, float]:
        """Run validation."""
        print("Running validation...")
        self.model.eval()
        
        val_metrics = {
            'val_loss': 0.0,
            'val_main_loss': 0.0,
            'val_mask_loss': 0.0,
            'val_sparsity': 0.0,
            'val_nm_compliance': 0.0
        }
        
        num_batches = 0
        
        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc="Validation"):
                if isinstance(batch, (list, tuple)):
                    batch = batch[0]
                
                # Compute losses
                main_loss, main_metrics = self.compute_diffusion_loss(batch)
                mask_loss, mask_metrics = self.compute_mask_loss(main_loss)
                
                val_metrics['val_main_loss'] += main_metrics['main_loss']
                val_metrics['val_mask_loss'] += mask_metrics['mask_loss']
                val_metrics['val_loss'] += main_loss.item() + mask_loss.item()
                
                num_batches += 1
        
        # Average metrics
        for key in val_metrics:
            val_metrics[key] /= max(num_batches, 1)
        
        # Sparsity metrics
        sparsity_summary = get_model_sparsity_summary(self.model)
        val_metrics['val_sparsity'] = sparsity_summary['overall_stats']['overall_sparsity']
        
        # N:M compliance
        nm_compliance_sum = 0
        nm_count = 0
        for module in self.model.modules():
            if isinstance(module, MaskProLayer):
                info = module.get_sparsity_info()
                nm_compliance_sum += info['nm_compliance']
                nm_count += 1
        
        if nm_count > 0:
            val_metrics['val_nm_compliance'] = nm_compliance_sum / nm_count
        
        self.model.train()
        return val_metrics
    
    def save_checkpoint(self, epoch: int, is_best: bool = False):
        """Save model checkpoint."""
        checkpoint_dir = os.path.join(
            self.config['output']['output_dir'],
            self.config['output']['checkpoints_dir']
        )
        
        checkpoint = {
            'epoch': epoch,
            'step': self.current_step,
            'model_state_dict': self.model.state_dict(),
            'model_optimizer_state_dict': self.optimizers['model'].state_dict(),
            'mask_optimizer_state_dict': self.optimizers['mask'].state_dict(),
            'model_scheduler_state_dict': self.lr_schedulers['model'].state_dict(),
            'mask_scheduler_state_dict': self.lr_schedulers['mask'].state_dict(),
            'config': self.config,
            'best_val_loss': self.best_val_loss,
            'training_logs': self.training_logs
        }
        
        # Save regular checkpoint
        checkpoint_path = os.path.join(checkpoint_dir, f"checkpoint_epoch_{epoch}.pt")
        torch.save(checkpoint, checkpoint_path)
        
        # Save best checkpoint
        if is_best:
            best_path = os.path.join(checkpoint_dir, "best_checkpoint.pt")
            torch.save(checkpoint, best_path)
            print(f"✓ Saved best checkpoint: {best_path}")
        
        # Save MaskPro specific state
        maskpro_path = os.path.join(checkpoint_dir, f"maskpro_state_epoch_{epoch}.pt")
        save_maskpro_state(self.model, maskpro_path, include_model_weights=False)
        
        print(f"✓ Saved checkpoint: {checkpoint_path}")
        
        # Cleanup old checkpoints
        self._cleanup_old_checkpoints(checkpoint_dir)
    
    def _cleanup_old_checkpoints(self, checkpoint_dir: str):
        """Keep only the last N checkpoints."""
        save_last_n = self.config['output'].get('save_last_n', 3)
        
        # Find all checkpoint files
        checkpoint_files = []
        for f in os.listdir(checkpoint_dir):
            if f.startswith("checkpoint_epoch_") and f.endswith(".pt"):
                epoch_num = int(f.split("_")[2].split(".")[0])
                checkpoint_files.append((epoch_num, f))
        
        # Sort by epoch and remove old ones
        checkpoint_files.sort(key=lambda x: x[0])
        
        if len(checkpoint_files) > save_last_n:
            for epoch_num, filename in checkpoint_files[:-save_last_n]:
                old_path = os.path.join(checkpoint_dir, filename)
                os.remove(old_path)
                
                # Also remove corresponding MaskPro state
                maskpro_file = f"maskpro_state_epoch_{epoch_num}.pt"
                maskpro_path = os.path.join(checkpoint_dir, maskpro_file)
                if os.path.exists(maskpro_path):
                    os.remove(maskpro_path)
    
    def train(self):
        """Main training loop."""
        print("=" * 60)
        print("🚀 Starting Diffusion MaskPro Training")
        print("=" * 60)
        
        self.train_start_time = time.time()
        
        # Setup all components
        self.load_model_and_scheduler()
        self.apply_maskpro_wrappers()
        self.setup_data_loaders()
        self.setup_optimizers()
        
        # Training loop
        training_config = self.config['training']
        
        for epoch in range(training_config['epochs']):
            self.current_epoch = epoch
            print(f"\n--- Epoch {epoch + 1}/{training_config['epochs']} ---")
            
            # Training
            epoch_metrics = self._train_epoch()
            
            # Validation
            if (epoch + 1) % training_config['val_freq'] == 0:
                val_metrics = self.validate()
                self.log_metrics(val_metrics, self.current_step, "val")
                
                # Check for best model
                current_val_loss = val_metrics['val_loss']
                if current_val_loss < self.best_val_loss:
                    self.best_val_loss = current_val_loss
                    self.save_checkpoint(epoch, is_best=True)
            
            # Save checkpoint
            if (epoch + 1) % training_config['save_freq'] == 0:
                self.save_checkpoint(epoch)
            
            # Log epoch summary
            print(f"Epoch {epoch + 1} completed - Loss: {epoch_metrics['avg_loss']:.6f}")
        
        # Final save
        self.save_checkpoint(training_config['epochs'] - 1)
        
        # Training completed
        total_time = time.time() - self.train_start_time
        print(f"\n✅ Training completed in {total_time / 3600:.2f} hours")
        
        if self.swanlab_run:
            self.swanlab_run.finish()
    
    def _train_epoch(self) -> Dict[str, float]:
        """Train for one epoch."""
        self.model.train()
        
        epoch_metrics = {
            'total_loss': 0.0,
            'main_loss': 0.0,
            'mask_loss': 0.0,
            'count': 0
        }
        
        progress_bar = tqdm(self.train_loader, desc=f"Epoch {self.current_epoch + 1}")
        
        for batch_idx, batch in enumerate(progress_bar):
            step_metrics = self.training_step(batch)
            
            # Accumulate metrics
            epoch_metrics['total_loss'] += step_metrics['total_loss']
            epoch_metrics['main_loss'] += step_metrics['main_loss']
            epoch_metrics['mask_loss'] += step_metrics['mask_loss']
            epoch_metrics['count'] += 1
            
            # Log step metrics
            self.log_metrics(step_metrics, self.current_step)
            
            # Update progress bar
            progress_bar.set_postfix({
                'Loss': f"{step_metrics['total_loss']:.4f}",
                'Sparsity': f"{step_metrics['sparsity_ratio']:.1%}",
                'N:M': f"{step_metrics.get('nm_compliance', 0):.1%}"
            })
            
            self.current_step += 1
        
        # Calculate average metrics
        avg_metrics = {
            'avg_loss': epoch_metrics['total_loss'] / epoch_metrics['count'],
            'avg_main_loss': epoch_metrics['main_loss'] / epoch_metrics['count'],
            'avg_mask_loss': epoch_metrics['mask_loss'] / epoch_metrics['count']
        }
        
        return avg_metrics 