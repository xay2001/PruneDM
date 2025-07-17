#!/bin/bash

# Wanda-Diff快速剪枝脚本 - 用于快速测试和验证
# 使用方法: bash scripts/prune/prune_ddpm_wanda_fast_cifar10.sh 0.3
# 参数 $1: 剪枝比率 (例如 0.3 表示30%剪枝)

echo "🚀 启动快速Wanda-Diff剪枝测试..."

python ddpm_prune.py \
--dataset cifar10 \
--model_path pretrained/ddpm_ema_cifar10 \
--save_path run/pruned/wanda-diff-fast/ddpm_cifar10_pruned \
--pruning_ratio ${1:-0.3} \
--batch_size 32 \
--pruner wanda-diff \
--device cuda:0 \
--wanda_calib_steps 256 \
--wanda_time_strategy mean \
--wanda_target_steps middle \
--wanda_activation_strategy mean \
--wanda_analyze_activations \
--wanda_save_analysis "run/pruned/wanda-diff-fast/ddpm_cifar10_pruned/activation_analysis.png"

echo "✅ 快速Wanda-Diff剪枝完成！" 