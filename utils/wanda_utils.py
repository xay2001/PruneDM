import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict


def compute_channel_importance(weight: torch.Tensor, activation_norms: torch.Tensor) -> torch.Tensor:
    """
    计算输出通道的重要性得分
    
    基于Wanda算法：S_channel(i) = Σ(j=0 to C_in-1) [||W_ij||_F * ||X_j||_2]
    
    Args:
        weight: 卷积层权重张量，形状为 (C_out, C_in, K, K)
        activation_norms: 输入激活的L2范数，形状为 (C_in,)
        
    Returns:
        torch.Tensor: 每个输出通道的重要性得分，形状为 (C_out,)
    """
    if weight.dim() != 4:
        raise ValueError(f"权重张量应该是4维的 (C_out, C_in, K, K)，但得到了 {weight.dim()} 维")
    
    if activation_norms.dim() != 1:
        raise ValueError(f"激活范数应该是1维的 (C_in,)，但得到了 {activation_norms.dim()} 维")
    
    c_out, c_in, k, k = weight.shape
    
    if c_in != activation_norms.shape[0]:
        raise ValueError(f"权重的输入通道数 {c_in} 与激活范数的长度 {activation_norms.shape[0]} 不匹配")
    
    # 计算每个输出通道的重要性
    importance_scores = torch.zeros(c_out, device=weight.device, dtype=weight.dtype)
    
    for i in range(c_out):
        # 对于输出通道i，计算所有输入连接的重要性
        channel_score = 0.0
        for j in range(c_in):
            # 计算权重核的Frobenius范数
            weight_norm = torch.norm(weight[i, j, :, :], p='fro')
            # 乘以对应的激活范数
            channel_score += weight_norm * activation_norms[j]
        
        importance_scores[i] = channel_score
    
    return importance_scores


def aggregate_activations(activation_lists: Dict[nn.Module, List[torch.Tensor]], 
                         strategy: str = 'mean',
                         time_range: Optional[Tuple[int, int]] = None) -> Dict[nn.Module, torch.Tensor]:
    """
    聚合多个时间步的激活数据
    
    Args:
        activation_lists: 每个模块的激活范数列表
        strategy: 聚合策略 ('mean', 'max', 'median', 'weighted_mean')
        time_range: 如果指定，只使用指定范围内的时间步
        
    Returns:
        Dict[nn.Module, torch.Tensor]: 聚合后的激活范数
    """
    aggregated = {}
    
    for module, norms_list in activation_lists.items():
        if not norms_list:
            continue
            
        # 选择时间步范围
        if time_range is not None:
            start_idx, end_idx = time_range
            norms_list = norms_list[start_idx:end_idx]
        
        if not norms_list:
            continue
            
        # 转换为张量
        norms_tensor = torch.stack(norms_list)  # (num_steps, C_in)
        
        # 根据策略聚合
        if strategy == 'mean':
            aggregated[module] = norms_tensor.mean(dim=0)
        elif strategy == 'max':
            aggregated[module] = norms_tensor.max(dim=0)[0]
        elif strategy == 'median':
            aggregated[module] = norms_tensor.median(dim=0)[0]
        elif strategy == 'weighted_mean':
            # 给后期时间步更高的权重（去噪末期更重要）
            weights = torch.linspace(0.5, 1.5, norms_tensor.shape[0], device=norms_tensor.device)
            weights = weights.unsqueeze(1)  # (num_steps, 1)
            weighted_sum = (norms_tensor * weights).sum(dim=0)
            weight_sum = weights.sum()
            aggregated[module] = weighted_sum / weight_sum
        else:
            raise ValueError(f"未知的聚合策略: {strategy}")
    
    return aggregated


def analyze_activation_distribution(activation_data: Dict[nn.Module, torch.Tensor], 
                                  save_path: Optional[str] = None) -> Dict[str, float]:
    """
    分析激活分布，检测是否存在"涌现的大幅值特征"
    
    Args:
        activation_data: 每个模块的聚合激活范数
        save_path: 如果指定，保存分析图表的路径
        
    Returns:
        Dict[str, float]: 分析结果统计
    """
    all_norms = []
    layer_stats = {}
    
    # 收集所有激活范数
    for module, norms in activation_data.items():
        all_norms.extend(norms.cpu().numpy().tolist())
        
        # 每层统计
        layer_stats[str(module)] = {
            'mean': norms.mean().item(),
            'std': norms.std().item(),
            'max': norms.max().item(),
            'min': norms.min().item(),
            'outlier_ratio': ((norms > norms.mean() + 2 * norms.std()).sum() / len(norms)).item()
        }
    
    all_norms = np.array(all_norms)
    
    # 全局统计
    global_stats = {
        'total_activations': len(all_norms),
        'global_mean': np.mean(all_norms),
        'global_std': np.std(all_norms),
        'global_max': np.max(all_norms),
        'global_min': np.min(all_norms),
        'skewness': float(np.mean(((all_norms - np.mean(all_norms)) / np.std(all_norms)) ** 3)),
        'kurtosis': float(np.mean(((all_norms - np.mean(all_norms)) / np.std(all_norms)) ** 4)),
        'outlier_ratio_2std': np.sum(np.abs(all_norms - np.mean(all_norms)) > 2 * np.std(all_norms)) / len(all_norms),
        'outlier_ratio_3std': np.sum(np.abs(all_norms - np.mean(all_norms)) > 3 * np.std(all_norms)) / len(all_norms)
    }
    
    # 可视化分析
    if save_path:
        plt.figure(figsize=(15, 10))
        
        # 激活范数分布直方图
        plt.subplot(2, 3, 1)
        plt.hist(all_norms, bins=50, alpha=0.7, edgecolor='black')
        plt.xlabel('Activation Norm')
        plt.ylabel('Frequency')
        plt.title('Distribution of Activation Norms')
        plt.yscale('log')
        
        # 对数尺度直方图
        plt.subplot(2, 3, 2)
        plt.hist(np.log(all_norms + 1e-8), bins=50, alpha=0.7, edgecolor='black')
        plt.xlabel('Log(Activation Norm)')
        plt.ylabel('Frequency')
        plt.title('Log-scale Distribution')
        
        # Box plot
        plt.subplot(2, 3, 3)
        plt.boxplot(all_norms)
        plt.ylabel('Activation Norm')
        plt.title('Box Plot of Activation Norms')
        plt.yscale('log')
        
        # 各层统计
        plt.subplot(2, 3, 4)
        layer_means = [stats['mean'] for stats in layer_stats.values()]
        layer_stds = [stats['std'] for stats in layer_stats.values()]
        layer_indices = range(len(layer_means))
        
        plt.errorbar(layer_indices, layer_means, yerr=layer_stds, fmt='o-')
        plt.xlabel('Layer Index')
        plt.ylabel('Mean Activation Norm')
        plt.title('Activation Norms by Layer')
        
        # 离群值比率
        plt.subplot(2, 3, 5)
        outlier_ratios = [stats['outlier_ratio'] for stats in layer_stats.values()]
        plt.plot(layer_indices, outlier_ratios, 'o-')
        plt.xlabel('Layer Index')
        plt.ylabel('Outlier Ratio (>2σ)')
        plt.title('Outlier Ratio by Layer')
        
        # QQ plot for normality test
        plt.subplot(2, 3, 6)
        from scipy import stats as scipy_stats
        scipy_stats.probplot(all_norms, dist="norm", plot=plt)
        plt.title('Q-Q Plot (Normal Distribution)')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    return global_stats


def parse_time_step_range(range_str: str, total_timesteps: int = 1000) -> Tuple[int, int]:
    """
    解析时间步范围字符串
    
    Args:
        range_str: 时间步范围字符串，格式如 "100-500" 或 "all" 或 "early" 或 "late"
        total_timesteps: 总时间步数
        
    Returns:
        Tuple[int, int]: (start_index, end_index)
    """
    if range_str.lower() == 'all':
        return (0, total_timesteps)
    elif range_str.lower() == 'early':
        return (int(total_timesteps * 0.7), total_timesteps)  # 高噪声阶段
    elif range_str.lower() == 'late':
        return (0, int(total_timesteps * 0.3))  # 低噪声阶段
    elif range_str.lower() == 'middle':
        start = int(total_timesteps * 0.3)
        end = int(total_timesteps * 0.7)
        return (start, end)
    elif '-' in range_str:
        try:
            start_str, end_str = range_str.split('-')
            start = int(start_str.strip())
            end = int(end_str.strip())
            return (start, end)
        except ValueError:
            raise ValueError(f"无法解析时间步范围: {range_str}")
    else:
        raise ValueError(f"无效的时间步范围格式: {range_str}")


def get_pruning_indices(importance_scores: torch.Tensor, pruning_ratio: float) -> List[int]:
    """
    根据重要性得分获取需要剪枝的通道索引
    
    Args:
        importance_scores: 重要性得分张量
        pruning_ratio: 剪枝比率 (0.0 - 1.0)
        
    Returns:
        List[int]: 需要剪枝的通道索引列表
    """
    num_channels = len(importance_scores)
    num_to_prune = int(num_channels * pruning_ratio)
    
    if num_to_prune == 0:
        return []
    
    # 获取重要性最低的通道索引
    _, indices = torch.topk(importance_scores, num_to_prune, largest=False)
    return indices.tolist()


def validate_pruning_compatibility(model: nn.Module, pruning_plan: Dict) -> bool:
    """
    验证剪枝计划的兼容性
    
    Args:
        model: 要剪枝的模型
        pruning_plan: 剪枝计划字典
        
    Returns:
        bool: 是否兼容
    """
    # 这里可以添加更多的验证逻辑
    # 例如检查跳跃连接的维度匹配等
    
    for module, indices in pruning_plan.items():
        if hasattr(module, 'out_channels'):
            if max(indices) >= module.out_channels:
                return False
    
    return True


def print_pruning_summary(pruning_plan: Dict[nn.Module, List[int]], 
                         importance_scores: Dict[nn.Module, torch.Tensor]) -> None:
    """
    打印剪枝摘要信息
    
    Args:
        pruning_plan: 剪枝计划
        importance_scores: 重要性得分
    """
    print("\n" + "="*60)
    print("Wanda-Diff 剪枝摘要")
    print("="*60)
    
    total_original = 0
    total_pruned = 0
    
    for i, (module, indices) in enumerate(pruning_plan.items()):
        if hasattr(module, 'out_channels'):
            original_channels = module.out_channels
            pruned_channels = len(indices)
            remaining_channels = original_channels - pruned_channels
            pruning_ratio = pruned_channels / original_channels
            
            total_original += original_channels
            total_pruned += pruned_channels
            
            # 获取该层的重要性统计
            if module in importance_scores:
                scores = importance_scores[module]
                min_score = scores.min().item()
                max_score = scores.max().item()
                mean_score = scores.mean().item()
                
                print(f"Layer {i+1:2d}: {original_channels:3d} -> {remaining_channels:3d} 通道 "
                      f"(剪枝 {pruning_ratio:.1%}) | 重要性: {min_score:.3f} - {max_score:.3f} (均值: {mean_score:.3f})")
    
    overall_ratio = total_pruned / total_original if total_original > 0 else 0
    print("-" * 60)
    print(f"总计: {total_original} -> {total_original - total_pruned} 通道 (剪枝 {overall_ratio:.1%})")
    print("="*60) 


class WandaImportance:
    """
    Wanda重要性评估器，兼容torch-pruning框架
    结合权重幅度和激活分布来计算通道重要性
    """
    
    def __init__(self, aggregated_activations, p=2, normalizer='mean'):
        """
        Args:
            aggregated_activations: 预计算的聚合激活数据 {layer: activation_norms}
            p: 范数类型 (默认L2范数)
            normalizer: 归一化方法
        """
        self.aggregated_activations = aggregated_activations
        self.p = p
        self.normalizer = normalizer
    
    def _normalize(self, importance, normalizer):
        """归一化重要性分数"""
        if normalizer is None:
            return importance
        elif normalizer == "sum":
            return importance / importance.sum()
        elif normalizer == "mean":
            return importance / importance.mean()
        elif normalizer == "max": 
            return importance / importance.max()
        elif normalizer == "gaussian":
            return (importance - importance.mean()) / (importance.std() + 1e-8)
        else:
            return importance
    
    def __call__(self, group, ch_groups=1):
        """
        计算剪枝组的重要性分数
        
        Args:
            group: torch-pruning的剪枝组 [(dependency, indices), ...]
            ch_groups: 通道分组数 (GroupNorm约束)
            
        Returns:
            torch.Tensor: 重要性分数
        """
        # 找到Conv2d层
        conv_layer = None
        conv_indices = None
        
        for dep, indices in group:
            layer = dep.target.module
            if isinstance(layer, torch.nn.Conv2d):
                conv_layer = layer
                conv_indices = indices
                break
        
        if conv_layer is None or conv_layer not in self.aggregated_activations:
            # 如果没有找到对应层或激活数据，使用权重幅度
            return self._fallback_magnitude_importance(group, ch_groups)
        
        # 获取权重和激活
        weight = conv_layer.weight.detach().cpu().float()  # (C_out, C_in, K, K)
        activation_norms = self.aggregated_activations[conv_layer].float()  # (C_in,)
        
        # 检查维度匹配
        if weight.shape[1] != activation_norms.shape[0]:
            return self._fallback_magnitude_importance(group, ch_groups)
        
        # 计算Wanda重要性: 权重范数 * 激活范数
        weight_norms = weight.abs().pow(self.p).sum(dim=[2, 3])  # (C_out, C_in)
        
        # 按输出通道计算重要性
        channel_importance = []
        for out_ch in conv_indices:
            if out_ch < weight_norms.shape[0]:
                # 该输出通道对所有输入通道的重要性
                ch_weight_norms = weight_norms[out_ch]  # (C_in,)
                # Wanda公式: |W| * |A|
                wanda_score = (ch_weight_norms * activation_norms).sum()
                channel_importance.append(wanda_score)
            else:
                channel_importance.append(0.0)
        
        importance = torch.tensor(channel_importance, dtype=torch.float32)
        
        # 处理通道分组约束
        if ch_groups > 1:
            group_size = len(importance) // ch_groups
            importance = importance[:group_size * ch_groups]  # 确保能整除
            importance = importance.view(ch_groups, group_size).mean(dim=0)
            importance = importance.repeat(ch_groups)
        
        # 归一化
        importance = self._normalize(importance, self.normalizer)
        
        return importance
    
    def _fallback_magnitude_importance(self, group, ch_groups=1):
        """回退到权重幅度重要性"""
        importance_scores = []
        
        for dep, indices in group:
            layer = dep.target.module
            if isinstance(layer, torch.nn.Conv2d):
                weight = layer.weight.detach()
                weight_norms = weight.abs().pow(self.p).sum(dim=[1, 2, 3])  # (C_out,)
                for idx in indices:
                    if idx < len(weight_norms):
                        importance_scores.append(weight_norms[idx].item())
                    else:
                        importance_scores.append(0.0)
                break
        
        if not importance_scores:
            # 最后的回退方案：随机重要性
            return torch.rand(len(group[0][1]))
        
        importance = torch.tensor(importance_scores, dtype=torch.float32)
        
        # 处理通道分组
        if ch_groups > 1:
            group_size = len(importance) // ch_groups
            importance = importance[:group_size * ch_groups]
            importance = importance.view(ch_groups, group_size).mean(dim=0)
            importance = importance.repeat(ch_groups)
        
        return self._normalize(importance, self.normalizer) 


def get_pruning_indices_with_groupnorm_constraint(
    importance_scores: torch.Tensor, 
    pruning_ratio: float, 
    num_groups: int = 32
) -> List[int]:
    """
    根据重要性得分获取需要剪枝的通道索引，确保剩余通道数满足GroupNorm约束
    
    Args:
        importance_scores: 重要性得分张量
        pruning_ratio: 目标剪枝比率 (0.0 - 1.0)
        num_groups: GroupNorm的分组数（默认32）
        
    Returns:
        List[int]: 需要剪枝的通道索引列表，确保剩余通道数能被num_groups整除
    """
    num_channels = len(importance_scores)
    target_prune = int(num_channels * pruning_ratio)
    
    # 确保剩余通道数能被num_groups整除
    remaining = num_channels - target_prune
    if remaining % num_groups != 0:
        # 调整到最近的满足约束的值（向下取整到最近的倍数）
        remaining = (remaining // num_groups) * num_groups
        # 确保至少保留一个分组
        if remaining == 0:
            remaining = num_groups
        target_prune = num_channels - remaining
    
    # 边界检查
    if target_prune <= 0:
        return []
    if target_prune >= num_channels:
        target_prune = num_channels - num_groups  # 至少保留一个分组
    
    # 获取重要性最低的通道索引
    _, indices = torch.topk(importance_scores, target_prune, largest=False)
    
    print(f"GroupNorm约束调整: {num_channels} -> {remaining} 通道 "
          f"(原计划剪枝{int(num_channels * pruning_ratio)}，"
          f"调整为剪枝{target_prune}，确保能被{num_groups}整除)")
    
    return indices.tolist() 