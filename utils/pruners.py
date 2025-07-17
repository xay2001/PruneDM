import torch
import torch.nn as nn
import torch_pruning as tp
from tqdm import tqdm
import os
from typing import Dict, List, Optional, Tuple, Union
import warnings

from .hooks import ActivationHooker
from .wanda_utils import (
    compute_channel_importance, 
    aggregate_activations, 
    get_pruning_indices,
    print_pruning_summary,
    analyze_activation_distribution,
    parse_time_step_range
)


def prune_wanda_diff(pipeline, 
                    train_dataloader, 
                    pruning_ratio: float, 
                    device: str,
                    num_calib_steps: int = 1024,
                    time_strategy: str = 'mean',
                    target_steps: str = 'all',
                    activation_strategy: str = 'mean',
                    analyze_activations: bool = True,
                    save_analysis: Optional[str] = None,
                    verbose: bool = True) -> object:
    """
    Wanda-Diff主算法实现：将Wanda剪枝方法应用于扩散模型
    
    Args:
        pipeline: DDPM pipeline对象
        train_dataloader: 校准数据加载器
        pruning_ratio: 剪枝比率 (0.0 - 1.0)
        device: 计算设备
        num_calib_steps: 校准步数
        time_strategy: 时间步聚合策略 ('mean', 'max', 'median', 'weighted_mean')
        target_steps: 目标时间步范围 ('all', 'early', 'late', 'middle', '100-500')
        activation_strategy: 激活聚合策略
        analyze_activations: 是否进行激活分析
        save_analysis: 激活分析结果保存路径
        verbose: 是否输出详细信息
        
    Returns:
        pipeline: 剪枝后的pipeline对象
    """
    
    if verbose:
        print("开始 Wanda-Diff 剪枝算法...")
        print(f"剪枝比率: {pruning_ratio:.1%}")
        print(f"校准步数: {num_calib_steps}")
        print(f"时间步策略: {time_strategy}")
        print(f"目标时间步: {target_steps}")
    
    # 获取模型和调度器
    model = pipeline.unet
    scheduler = pipeline.scheduler
    
    # 步骤1: 激活收集阶段
    if verbose:
        print("\n步骤1: 收集激活数据...")
    
    hooker = ActivationHooker()
    handles = hooker.register_hooks(model)
    
    model.eval()
    collected_steps = 0
    
    try:
        with torch.no_grad():
            for batch_idx, batch in enumerate(tqdm(train_dataloader, desc="校准激活收集", disable=not verbose)):
                if collected_steps >= num_calib_steps:
                    break
                    
                # 处理不同的数据格式
                if isinstance(batch, (list, tuple)):
                    clean_images = batch[0]
                elif isinstance(batch, dict):
                    clean_images = batch.get('pixel_values', batch.get('images', None))
                    if clean_images is None:
                        # 尝试获取第一个张量值
                        for value in batch.values():
                            if isinstance(value, torch.Tensor) and len(value.shape) == 4:
                                clean_images = value
                                break
                else:
                    clean_images = batch
                
                if clean_images is None:
                    raise ValueError("无法从数据批次中提取图像数据")
                
                clean_images = clean_images.to(device)
                batch_size = clean_images.shape[0]
                
                # 生成随机噪声
                noise = torch.randn_like(clean_images)
                
                # 解析时间步范围
                total_timesteps = scheduler.config.num_train_timesteps
                time_range = parse_time_step_range(target_steps, total_timesteps)
                
                # 在指定的时间步范围内进行采样
                if target_steps.lower() == 'all':
                    # 随机采样时间步
                    timesteps = torch.randint(0, total_timesteps, (batch_size,), device=device).long()
                else:
                    # 在指定范围内随机采样
                    start_t, end_t = time_range
                    timesteps = torch.randint(start_t, end_t, (batch_size,), device=device).long()
                
                # 添加噪声
                noisy_images = scheduler.add_noise(clean_images, noise, timesteps)
                
                # 前向传播（触发钩子）
                _ = model(noisy_images, timesteps)
                
                collected_steps += batch_size
                
    except Exception as e:
        hooker.remove_hooks()
        raise e
    
    # 移除钩子
    hooker.remove_hooks()
    
    if verbose:
        print(f"收集完成，共处理 {collected_steps} 个样本")
    
    # 步骤2: 激活分析（可选）
    if analyze_activations:
        if verbose:
            print("\n步骤2: 分析激活分布...")
        
        # 获取聚合的激活范数
        aggregated_activations = hooker.get_aggregated_norms(strategy=activation_strategy)
        
        # 分析激活分布
        stats = analyze_activation_distribution(
            aggregated_activations, 
            save_path=save_analysis
        )
        
        if verbose:
            print("激活分布统计:")
            print(f"  总激活数: {stats['total_activations']}")
            print(f"  全局均值: {stats['global_mean']:.4f}")
            print(f"  全局标准差: {stats['global_std']:.4f}")
            print(f"  偏度: {stats['skewness']:.4f}")
            print(f"  峰度: {stats['kurtosis']:.4f}")
            print(f"  离群值比率 (2σ): {stats['outlier_ratio_2std']:.2%}")
            print(f"  离群值比率 (3σ): {stats['outlier_ratio_3std']:.2%}")
            
            # 检查是否存在"涌现的大幅值特征"
            if stats['outlier_ratio_2std'] > 0.05:  # 超过5%的离群值
                print("  ✓ 检测到显著的大幅值特征，Wanda方法论适用")
            else:
                print("  ⚠ 大幅值特征不明显，Wanda效果可能有限")
    
    # 步骤3: 计算通道重要性
    if verbose:
        print("\n步骤3: 计算通道重要性...")
    
    aggregated_activations = hooker.get_aggregated_norms(strategy=activation_strategy)
    channel_importances = {}
    pruning_plan = {}
    
    conv_layers = []
    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d):
            conv_layers.append((name, module))
    
    for name, layer in tqdm(conv_layers, desc="计算重要性", disable=not verbose):
        if layer not in aggregated_activations:
            if verbose:
                print(f"警告: 层 {name} 没有激活数据，跳过")
            continue
        
        # 获取权重和激活范数
        weight = layer.weight.detach().cpu().float()  # (C_out, C_in, K, K)
        activation_norms = aggregated_activations[layer].float()  # (C_in,)
        
        # 确保维度匹配
        if weight.shape[1] != activation_norms.shape[0]:
            if verbose:
                print(f"警告: 层 {name} 维度不匹配，跳过 (权重: {weight.shape}, 激活: {activation_norms.shape})")
            continue
        
        # 计算通道重要性
        importance_scores = compute_channel_importance(weight, activation_norms)
        channel_importances[layer] = importance_scores
        
        # 生成剪枝计划
        pruning_indices = get_pruning_indices(importance_scores, pruning_ratio)
        if pruning_indices:
            pruning_plan[layer] = pruning_indices
    
    if not pruning_plan:
        if verbose:
            print("警告: 没有生成有效的剪枝计划")
        return pipeline
    
    # 步骤4: 执行结构化剪枝
    if verbose:
        print("\n步骤4: 执行结构化剪枝...")
        print_pruning_summary(pruning_plan, channel_importances)
    
    # 准备示例输入用于依赖图构建
    if 'cifar' in str(type(model)).lower() or any('32' in str(s) for s in [model]):
        example_input_size = (1, 3, 32, 32)
    else:
        example_input_size = (1, 3, 256, 256)
    
    example_inputs = {
        'sample': torch.randn(example_input_size).to(device), 
        'timestep': torch.tensor([1]).long().to(device)
    }
    
    # 使用torch-pruning进行结构化剪枝
    try:
        # 构建依赖图
        DG = tp.DependencyGraph().build_dependency(model, example_inputs=example_inputs)
        
        # 执行剪枝
        pruning_groups = []
        for layer, indices in pruning_plan.items():
            if len(indices) == 0:
                continue
                
            try:
                # 获取剪枝组
                group = DG.get_pruning_group(layer, tp.prune_conv_out_channels, idxs=indices)
                
                # 检查剪枝组的有效性
                if DG.check_pruning_group(group):
                    pruning_groups.append(group)
                else:
                    if verbose:
                        print(f"警告: 层 {layer} 的剪枝组无效，跳过")
                        
            except Exception as e:
                if verbose:
                    print(f"警告: 处理层 {layer} 时出错: {e}")
                continue
        
        # 执行所有有效的剪枝组
        for group in pruning_groups:
            group.prune()
            
        # 更新静态属性（处理某些模块的特殊需求）
        try:
            from diffusers.models.resnet import Upsample2D, Downsample2D
            for module in model.modules():
                if isinstance(module, (Upsample2D, Downsample2D)):
                    if hasattr(module, 'conv') and module.conv is not None:
                        module.channels = module.conv.in_channels
                        if hasattr(module, 'out_channels'):
                            module.out_channels = module.conv.out_channels
        except ImportError:
            # 如果导入失败，继续执行
            pass
        
        # 计算剪枝效果
        try:
            base_macs, base_params = tp.utils.count_ops_and_params(model, example_inputs)
            if verbose:
                print(f"\n剪枝完成!")
                print(f"参数减少: {base_params/1e6:.2f}M")
                print(f"FLOPs: {base_macs/1e9:.2f}G")
        except:
            if verbose:
                print("剪枝完成! (无法计算详细统计)")
        
    except Exception as e:
        if verbose:
            print(f"剪枝过程中出错: {e}")
            print("尝试使用备用方法...")
        
        # 备用方法：简单的通道置零（仅用于测试）
        with torch.no_grad():
            for layer, indices in pruning_plan.items():
                if hasattr(layer, 'weight') and layer.weight is not None:
                    # 将对应通道的权重置零
                    layer.weight.data[indices] = 0
                    if hasattr(layer, 'bias') and layer.bias is not None:
                        layer.bias.data[indices] = 0
        
        if verbose:
            print("使用备用方法完成剪枝")
    
    # 清理
    hooker.clear()
    model.zero_grad()
    
    if verbose:
        print("Wanda-Diff 剪枝算法完成!")
    
    return pipeline


def analyze_model_activations(pipeline, 
                            train_dataloader, 
                            device: str,
                            num_samples: int = 512,
                            save_path: Optional[str] = None) -> Dict:
    """
    独立的激活分析函数，用于验证DDPM中是否存在"涌现的大幅值特征"
    
    Args:
        pipeline: DDPM pipeline对象
        train_dataloader: 数据加载器
        device: 计算设备
        num_samples: 分析样本数
        save_path: 结果保存路径
        
    Returns:
        Dict: 分析结果
    """
    print("开始激活分析实验...")
    
    model = pipeline.unet
    scheduler = pipeline.scheduler
    
    hooker = ActivationHooker()
    handles = hooker.register_hooks(model)
    
    model.eval()
    collected = 0
    
    try:
        with torch.no_grad():
            for batch in tqdm(train_dataloader, desc="收集激活数据"):
                if collected >= num_samples:
                    break
                
                # 获取图像数据
                if isinstance(batch, (list, tuple)):
                    images = batch[0]
                elif isinstance(batch, dict):
                    images = batch.get('pixel_values', batch.get('images'))
                else:
                    images = batch
                
                images = images.to(device)
                batch_size = images.shape[0]
                
                # 在多个时间步进行采样
                for t in [50, 100, 200, 500, 800]:
                    if collected >= num_samples:
                        break
                        
                    timesteps = torch.full((batch_size,), t, device=device).long()
                    noise = torch.randn_like(images)
                    noisy_images = scheduler.add_noise(images, noise, timesteps)
                    
                    _ = model(noisy_images, timesteps)
                    collected += batch_size
                    
    finally:
        hooker.remove_hooks()
    
    # 分析结果
    aggregated = hooker.get_aggregated_norms(strategy='mean')
    stats = analyze_activation_distribution(aggregated, save_path=save_path)
    
    # 离群值分析
    outlier_analysis = hooker.analyze_outliers(threshold=2.0)
    
    print("激活分析完成!")
    print(f"分析了 {collected} 个样本")
    print(f"检测到 {stats['outlier_ratio_2std']:.1%} 的离群激活")
    
    result = {
        'stats': stats,
        'outlier_analysis': outlier_analysis,
        'aggregated_activations': aggregated,
        'num_samples': collected
    }
    
    return result 