from diffusers import DiffusionPipeline, DDPMPipeline, DDIMPipeline, DDIMScheduler, DDPMScheduler
from diffusers.models import UNet2DModel
import torch_pruning as tp
import torch
import torchvision
from torchvision import transforms
import torchvision
from tqdm import tqdm
import os
from glob import glob
from PIL import Image
import accelerate
import utils

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--dataset", type=str,  default=None, help="path to an image folder")
parser.add_argument("--model_path", type=str, required=True)
parser.add_argument("--save_path", type=str, required=True)
parser.add_argument("--pruning_ratio", type=float, default=0.3)
parser.add_argument("--batch_size", type=int, default=128)
parser.add_argument("--device", type=str, default='cpu')
parser.add_argument("--pruner", type=str, default='taylor', choices=['taylor', 'random', 'magnitude', 'reinit', 'diff-pruning', 'wanda-diff'])

parser.add_argument("--thr", type=float, default=0.05, help="threshold for diff-pruning")

# Wanda-Diff specific arguments
parser.add_argument("--wanda_calib_steps", type=int, default=1024, help="number of calibration steps for Wanda-Diff")
parser.add_argument("--wanda_time_strategy", type=str, default='mean', choices=['mean', 'max', 'median', 'weighted_mean'], help="time step aggregation strategy")
parser.add_argument("--wanda_target_steps", type=str, default='all', help="target time steps (all, early, late, middle, start-end)")
parser.add_argument("--wanda_activation_strategy", type=str, default='mean', choices=['mean', 'max', 'median'], help="activation aggregation strategy")
parser.add_argument("--wanda_analyze_activations", action='store_true', help="analyze activation distributions")
parser.add_argument("--wanda_save_analysis", type=str, default=None, help="path to save activation analysis plots")

args = parser.parse_args()

batch_size = args.batch_size
dataset = args.dataset

if __name__=='__main__':
    
    # loading images for gradient-based pruning and wanda-diff
    if args.pruner in ['taylor', 'diff-pruning', 'wanda-diff']:
        dataset_name = args.dataset if args.dataset else 'cifar10'  # default to cifar10 for wanda-diff
        dataset = utils.get_dataset(dataset_name)
        print(f"Dataset size: {len(dataset)}")
        train_dataloader = torch.utils.data.DataLoader(
            dataset, batch_size=args.batch_size, shuffle=True, num_workers=4, drop_last=True
        )
        
        if args.pruner in ['taylor', 'diff-pruning']:
            import torch_pruning as tp
            clean_images = next(iter(train_dataloader))
            if isinstance(clean_images, (list, tuple)):
                clean_images = clean_images[0]
            clean_images = clean_images.to(args.device)
            noise = torch.randn(clean_images.shape).to(clean_images.device)

    # Loading pretrained model
    print("Loading pretrained model from {}".format(args.model_path))
    pipeline = DDPMPipeline.from_pretrained(args.model_path).to(args.device)
    scheduler = pipeline.scheduler
    model = pipeline.unet.eval()
    if 'cifar' in args.model_path:
        example_inputs = {'sample': torch.randn(1, 3, 32, 32).to(args.device), 'timestep': torch.ones((1,)).long().to(args.device)}
    else:
        example_inputs = {'sample': torch.randn(1, 3, 256, 256).to(args.device), 'timestep': torch.ones((1,)).long().to(args.device)}

    if args.pruning_ratio>0:
        if args.pruner == 'wanda-diff':
            # Use Wanda-Diff algorithm
            from utils.pruners import prune_wanda_diff
            
            print("Starting Wanda-Diff pruning...")
            
            # Setup analysis path if requested
            analysis_path = None
            if args.wanda_analyze_activations:
                analysis_path = args.wanda_save_analysis
                if analysis_path is None:
                    analysis_path = os.path.join(args.save_path, "wanda_analysis.png")
                # Create directory if needed
                os.makedirs(os.path.dirname(analysis_path), exist_ok=True)
            
            # Apply Wanda-Diff pruning
            pipeline = prune_wanda_diff(
                pipeline=pipeline,
                train_dataloader=train_dataloader, 
                pruning_ratio=args.pruning_ratio,
                device=args.device,
                num_calib_steps=args.wanda_calib_steps,
                time_strategy=args.wanda_time_strategy,
                target_steps=args.wanda_target_steps,
                activation_strategy=args.wanda_activation_strategy,
                analyze_activations=args.wanda_analyze_activations,
                save_analysis=analysis_path,
                verbose=True
            )
            
            # Update model reference
            model = pipeline.unet
            
        elif args.pruner == 'taylor':
            imp = tp.importance.TaylorImportance(multivariable=True) # standard first-order taylor expansion
        elif args.pruner == 'random' or args.pruner=='reinit':
            imp = tp.importance.RandomImportance()
        elif args.pruner == 'magnitude':
            imp = tp.importance.MagnitudeImportance()
        elif args.pruner == 'diff-pruning':
            imp = tp.importance.TaylorImportance(multivariable=False) # a modified version, estimating the accumulated error of weight removal
        else:
            raise NotImplementedError

        # Traditional pruning methods (skip for wanda-diff)
        if args.pruner != 'wanda-diff':
            ignored_layers = [model.conv_out]
            channel_groups = {}
            #from diffusers.models.attention import 
            #for m in model.modules():
            #    if isinstance(m, AttentionBlock):
            #        channel_groups[m.query] = m.num_heads
            #        channel_groups[m.key] = m.num_heads
            #        channel_groups[m.value] = m.num_heads
            
            pruner = tp.pruner.MagnitudePruner(
                model,
                example_inputs,
                importance=imp,
                iterative_steps=1,
                channel_groups=channel_groups,
                ch_sparsity=args.pruning_ratio,
                ignored_layers=ignored_layers,
            )

            base_macs, base_params = tp.utils.count_ops_and_params(model, example_inputs)
            model.zero_grad()
            model.eval()
            import random

            if args.pruner in ['taylor', 'diff-pruning']:
                loss_max = 0
                print("Accumulating gradients for pruning...")
                for step_k in tqdm(range(1000)):
                    timesteps = (step_k*torch.ones((args.batch_size,), device=clean_images.device)).long()
                    noisy_images = scheduler.add_noise(clean_images, noise, timesteps)
                    model_output = model(noisy_images, timesteps).sample
                    loss = torch.nn.functional.mse_loss(model_output, noise) 
                    loss.backward() 
                    
                    if args.pruner=='diff-pruning':
                        if loss>loss_max: loss_max = loss
                        if loss<loss_max * args.thr: break # taylor expansion over pruned timesteps ( L_t / L_max > thr )

            for g in pruner.step(interactive=True):
                g.prune()

            # Update static attributes
            from diffusers.models.resnet import Upsample2D, Downsample2D
            for m in model.modules():
                if isinstance(m, (Upsample2D, Downsample2D)):
                    m.channels = m.conv.in_channels
                    m.out_channels == m.conv.out_channels

            macs, params = tp.utils.count_ops_and_params(model, example_inputs)
            print(model)
            print("#Params: {:.4f} M => {:.4f} M".format(base_params/1e6, params/1e6))
            print("#MACS: {:.4f} G => {:.4f} G".format(base_macs/1e9, macs/1e9))
            model.zero_grad()
            del pruner

            if args.pruner=='reinit':
                def reset_parameters(model):
                    for m in model.modules():
                        if hasattr(m, 'reset_parameters'):
                            m.reset_parameters()
                reset_parameters(model)

    pipeline.save_pretrained(args.save_path)
    if args.pruning_ratio>0:
        os.makedirs(os.path.join(args.save_path, "pruned"), exist_ok=True)
        torch.save(model, os.path.join(args.save_path, "pruned", "unet_pruned.pth"))

    # Sampling images from the pruned model
    pipeline = DDIMPipeline(
        unet = model,
        scheduler = DDIMScheduler.from_pretrained(args.save_path, subfolder="scheduler")
    )
    with torch.no_grad():
        generator = torch.Generator(device=pipeline.device).manual_seed(0)
        pipeline.to("cuda")
        images = pipeline(num_inference_steps=100, batch_size=args.batch_size, generator=generator, output_type="numpy").images
        os.makedirs(os.path.join(args.save_path, 'vis'), exist_ok=True)
        torchvision.utils.save_image(torch.from_numpy(images).permute([0, 3, 1, 2]), "{}/vis/after_pruning.png".format(args.save_path))
        
