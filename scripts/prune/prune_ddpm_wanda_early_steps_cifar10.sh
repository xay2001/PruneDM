#!/bin/bash

# Wanda-Diff剪枝脚本 - 关注早期时间步 (高噪声阶段)
# 使用方法: bash scripts/prune/prune_ddpm_wanda_early_steps_cifar10.sh 0.3
# 参数 $1: 剪枝比率 (例如 0.3 表示30%剪枝)

python ddpm_prune.py \
--dataset cifar10 \
--model_path pretrained/ddpm_ema_cifar10 \
--save_path run/pruned/wanda-diff-early/ddpm_cifar10_pruned \
--pruning_ratio ${1:-0.3} \
--batch_size 64 \
--pruner wanda-diff \
--device cuda:0 \
--wanda_calib_steps 1024 \
--wanda_time_strategy mean \
--wanda_target_steps early \
--wanda_activation_strategy mean \
--wanda_analyze_activations \
--wanda_save_analysis "run/pruned/wanda-diff-early/ddpm_cifar10_pruned/activation_analysis.png" 