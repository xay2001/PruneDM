#!/usr/bin/env python3
"""
DDPM激活分析实验
验证扩散模型中是否存在"涌现的大幅值特征"
"""

import argparse
import torch
import os
import sys
from diffusers import DDPMPipeline
sys.path.append('..')
import utils
from utils.pruners import analyze_model_activations


def main():
    parser = argparse.ArgumentParser(description='DDPM激活分析实验')
    parser.add_argument('--model_path', type=str, default='google/ddpm-cifar10-32',
                       help='预训练模型路径')
    parser.add_argument('--dataset', type=str, default='cifar10',
                       help='数据集名称')
    parser.add_argument('--device', type=str, default='cuda:0',
                       help='计算设备')
    parser.add_argument('--batch_size', type=int, default=32,
                       help='批次大小')
    parser.add_argument('--num_samples', type=int, default=1024,
                       help='分析样本数量')
    parser.add_argument('--save_path', type=str, default='./activation_analysis_results',
                       help='结果保存路径')
    
    args = parser.parse_args()
    
    print("="*60)
    print("DDPM 激活分析实验")
    print("="*60)
    print(f"模型: {args.model_path}")
    print(f"数据集: {args.dataset}")
    print(f"设备: {args.device}")
    print(f"样本数: {args.num_samples}")
    print("-"*60)
    
    # 创建输出目录
    os.makedirs(args.save_path, exist_ok=True)
    
    # 加载模型
    print("加载预训练模型...")
    pipeline = DDPMPipeline.from_pretrained(args.model_path).to(args.device)
    
    # 准备数据
    print("准备数据集...")
    dataset = utils.get_dataset(args.dataset)
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=args.batch_size, shuffle=True, 
        num_workers=4, drop_last=True
    )
    
    # 进行激活分析
    print("开始激活分析...")
    save_plot_path = os.path.join(args.save_path, "activation_distribution.png")
    
    results = analyze_model_activations(
        pipeline=pipeline,
        train_dataloader=dataloader,
        device=args.device,
        num_samples=args.num_samples,
        save_path=save_plot_path
    )
    
    # 保存详细结果
    stats = results['stats']
    outlier_analysis = results['outlier_analysis']
    
    # 生成报告
    report_path = os.path.join(args.save_path, "analysis_report.txt")
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("DDPM 激活分析报告\n")
        f.write("="*50 + "\n")
        f.write(f"模型: {args.model_path}\n")
        f.write(f"数据集: {args.dataset}\n")
        f.write(f"分析样本数: {results['num_samples']}\n\n")
        
        f.write("全局统计信息:\n")
        f.write("-"*30 + "\n")
        f.write(f"总激活数: {stats['total_activations']}\n")
        f.write(f"全局均值: {stats['global_mean']:.6f}\n")
        f.write(f"全局标准差: {stats['global_std']:.6f}\n")
        f.write(f"最大值: {stats['global_max']:.6f}\n")
        f.write(f"最小值: {stats['global_min']:.6f}\n")
        f.write(f"偏度: {stats['skewness']:.4f}\n")
        f.write(f"峰度: {stats['kurtosis']:.4f}\n")
        f.write(f"离群值比率 (2σ): {stats['outlier_ratio_2std']:.2%}\n")
        f.write(f"离群值比率 (3σ): {stats['outlier_ratio_3std']:.2%}\n\n")
        
        f.write("Wanda方法论适用性分析:\n")
        f.write("-"*30 + "\n")
        if stats['outlier_ratio_2std'] > 0.05:
            f.write("✓ 检测到显著的大幅值特征 (>5%离群值)\n")
            f.write("✓ Wanda方法论在此模型上应该有效\n")
            applicability = "适用"
        elif stats['outlier_ratio_2std'] > 0.02:
            f.write("△ 检测到中等程度的大幅值特征 (2-5%离群值)\n")
            f.write("△ Wanda方法论可能有一定效果\n")
            applicability = "可能适用"
        else:
            f.write("✗ 大幅值特征不明显 (<2%离群值)\n")
            f.write("✗ Wanda方法论效果可能有限\n")
            applicability = "不太适用"
        
        f.write(f"\n结论: Wanda-Diff方法论对该模型 {applicability}\n")
        
        # 各层详细分析
        f.write("\n各层激活分析:\n")
        f.write("-"*30 + "\n")
        for i, (module, analysis) in enumerate(outlier_analysis.items()):
            f.write(f"Layer {i+1}:\n")
            f.write(f"  总通道数: {analysis['total_channels']}\n")
            f.write(f"  离群通道数: {analysis['outlier_channels']}\n")
            f.write(f"  离群比率: {analysis['outlier_ratio']:.2%}\n")
            f.write(f"  均值: {analysis['mean_norm']:.6f}\n")
            f.write(f"  标准差: {analysis['std_norm']:.6f}\n")
            f.write(f"  最大值: {analysis['max_norm']:.6f}\n")
            f.write(f"  最小值: {analysis['min_norm']:.6f}\n\n")
    
    print("="*60)
    print("激活分析完成!")
    print(f"结果保存在: {args.save_path}")
    print(f"分析图表: {save_plot_path}")
    print(f"详细报告: {report_path}")
    print("="*60)
    
    # 打印关键结论
    print("\n关键发现:")
    print(f"• 离群值比率 (2σ): {stats['outlier_ratio_2std']:.2%}")
    if stats['outlier_ratio_2std'] > 0.05:
        print("• ✓ 检测到显著的涌现大幅值特征")
        print("• ✓ Wanda-Diff方法论应该有效")
    elif stats['outlier_ratio_2std'] > 0.02:
        print("• △ 检测到中等程度的涌现特征")
        print("• △ Wanda-Diff方法论可能有效")
    else:
        print("• ✗ 涌现特征不明显")
        print("• ✗ Wanda-Diff方法论效果可能有限")
    
    return results


if __name__ == '__main__':
    main() 