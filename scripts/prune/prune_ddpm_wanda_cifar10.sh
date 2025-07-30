#!/bin/bash

# Wanda-Diff剪枝脚本 for CIFAR-10 DDPM
# 使用方法: bash scripts/prune/prune_ddpm_wanda_cifar10.sh 0.3
# 参数 $1: 剪枝比率 (例如 0.3 表示30%剪枝)

CUDA_VISIBLE_DEVICES=1 python ddpm_prune.py \
--dataset cifar10 \
--model_path pretrained/ddpm_ema_cifar10 \
--save_path run/pruned/wanda-diff/ddpm_cifar10_pruned \
--pruning_ratio ${1:-0.3} \
--batch_size 64 \
--pruner wanda-diff \
--device cuda:0 \
--wanda_calib_steps 1024 \
--wanda_time_strategy mean \
--wanda_target_steps all \
--wanda_activation_strategy mean \
--wanda_analyze_activations \
--wanda_save_analysis "run/pruned/wanda-diff/ddpm_cifar10_pruned/activation_analysis.png" 