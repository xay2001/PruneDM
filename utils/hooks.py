import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple
from collections import defaultdict
import numpy as np
from tqdm import tqdm


class ActivationHooker:
    """
    管理U-Net激活数据收集的钩子类
    用于Wanda-Diff算法中收集卷积层的输入激活数据
    """
    
    def __init__(self):
        self.activations = defaultdict(list)  # module -> list of activation norms
        self.handles = []
        
    def hook_fn(self, module, input, output):
        """
        前向钩子函数，收集输入激活的L2范数
        
        Args:
            module: 当前模块（Conv2d层）
            input: 模块输入 (tuple)
            output: 模块输出
        """
        # 获取输入激活张量
        if isinstance(input, tuple):
            activation = input[0]
        else:
            activation = input
            
        if activation is None or not isinstance(activation, torch.Tensor):
            return
            
        # 计算每个输入通道的L2范数
        # activation shape: (N, C_in, H, W)
        if len(activation.shape) == 4:
            # 沿空间维度计算L2范数：(N, C_in, H, W) -> (N, C_in)
            norms = torch.norm(activation.float(), p=2, dim=(2, 3))
            # 沿batch维度取平均：(N, C_in) -> (C_in,)
            avg_norms = norms.mean(dim=0).detach().cpu()
            
            # 存储到CPU以节省GPU内存
            self.activations[module].append(avg_norms)
        
    def register_hooks(self, model):
        """
        在模型的所有Conv2d层注册前向钩子
        
        Args:
            model: U-Net模型
            
        Returns:
            List[RemovableHandle]: 钩子句柄列表
        """
        handles = []
        for name, module in model.named_modules():
            if isinstance(module, nn.Conv2d):
                handle = module.register_forward_hook(self.hook_fn)
                handles.append(handle)
                
        self.handles = handles
        return handles
    
    def remove_hooks(self, handles=None):
        """
        移除所有注册的钩子
        
        Args:
            handles: 钩子句柄列表，如果为None则使用self.handles
        """
        if handles is None:
            handles = self.handles
            
        for handle in handles:
            handle.remove()
        
        # 清理存储的句柄
        self.handles = []
    
    def get_aggregated_norms(self, strategy='mean'):
        """
        获取聚合后的激活范数
        
        Args:
            strategy: 聚合策略 ('mean', 'max', 'median')
            
        Returns:
            Dict[nn.Module, torch.Tensor]: 每个模块的聚合激活范数
        """
        aggregated = {}
        
        for module, norms_list in self.activations.items():
            if not norms_list:
                continue
                
            # 将列表转换为张量 (num_steps, C_in)
            norms_tensor = torch.stack(norms_list)
            
            # 根据策略聚合
            if strategy == 'mean':
                aggregated[module] = norms_tensor.mean(dim=0)
            elif strategy == 'max':
                aggregated[module] = norms_tensor.max(dim=0)[0]
            elif strategy == 'median':
                aggregated[module] = norms_tensor.median(dim=0)[0]
            else:
                raise ValueError(f"未知的聚合策略: {strategy}")
                
        return aggregated
    
    def clear(self):
        """清除所有收集的激活数据"""
        self.activations.clear()
        
    def get_statistics(self):
        """
        获取激活数据的统计信息
        
        Returns:
            Dict: 包含统计信息的字典
        """
        stats = {}
        
        for module, norms_list in self.activations.items():
            if not norms_list:
                continue
                
            # 计算基本统计量
            norms_tensor = torch.stack(norms_list)  # (num_steps, C_in)
            
            stats[module] = {
                'num_timesteps': len(norms_list),
                'num_channels': norms_tensor.shape[1],
                'mean_across_time': norms_tensor.mean().item(),
                'std_across_time': norms_tensor.std().item(),
                'max_across_time': norms_tensor.max().item(),
                'min_across_time': norms_tensor.min().item()
            }
            
        return stats
        
    def analyze_outliers(self, threshold=2.0):
        """
        分析激活中的离群值
        
        Args:
            threshold: 标准差阈值，超过此值被认为是离群值
            
        Returns:
            Dict: 离群值分析结果
        """
        outlier_analysis = {}
        
        for module, norms_list in self.activations.items():
            if not norms_list:
                continue
                
            # 聚合所有时间步的激活范数
            aggregated_norms = torch.stack(norms_list).mean(dim=0)  # (C_in,)
            
            # 计算统计量
            mean_norm = aggregated_norms.mean()
            std_norm = aggregated_norms.std()
            
            # 识别离群值
            outlier_mask = torch.abs(aggregated_norms - mean_norm) > threshold * std_norm
            outlier_indices = torch.where(outlier_mask)[0]
            
            outlier_analysis[module] = {
                'total_channels': len(aggregated_norms),
                'outlier_channels': len(outlier_indices),
                'outlier_ratio': len(outlier_indices) / len(aggregated_norms),
                'outlier_indices': outlier_indices.tolist(),
                'mean_norm': mean_norm.item(),
                'std_norm': std_norm.item(),
                'max_norm': aggregated_norms.max().item(),
                'min_norm': aggregated_norms.min().item()
            }
            
        return outlier_analysis 