#!/bin/bash

# Wanda-Diff配置对比脚本
# 使用方法: bash scripts/prune/compare_wanda_configurations.sh 0.3
# 参数 $1: 剪枝比率 (例如 0.3 表示30%剪枝)

PRUNING_RATIO=${1:-0.3}

echo "🔍 开始Wanda-Diff多配置对比测试..."
echo "剪枝比率: ${PRUNING_RATIO}"
echo "=================================="

# 1. 标准配置
echo "🎯 测试1: 标准Wanda-Diff配置..."
bash scripts/prune/prune_ddpm_wanda_cifar10.sh ${PRUNING_RATIO}

# 2. 早期时间步配置
echo "🎯 测试2: 早期时间步配置..."
bash scripts/prune/prune_ddpm_wanda_early_steps_cifar10.sh ${PRUNING_RATIO}

# 3. 加权平均配置
echo "🎯 测试3: 加权平均配置..."
bash scripts/prune/prune_ddpm_wanda_weighted_cifar10.sh ${PRUNING_RATIO}

# 4. 快速测试配置
echo "🎯 测试4: 快速测试配置..."
bash scripts/prune/prune_ddpm_wanda_fast_cifar10.sh ${PRUNING_RATIO}

echo "✅ 所有Wanda-Diff配置测试完成！"
echo "查看结果目录:"
echo "  - 标准配置: run/pruned/wanda-diff/ddpm_cifar10_pruned"
echo "  - 早期时间步: run/pruned/wanda-diff-early/ddpm_cifar10_pruned"
echo "  - 加权平均: run/pruned/wanda-diff-weighted/ddpm_cifar10_pruned"
echo "  - 快速测试: run/pruned/wanda-diff-fast/ddpm_cifar10_pruned" 