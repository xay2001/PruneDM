#!/bin/bash

# Wanda-Diff测试脚本
# 使用预训练的DDPM模型在CIFAR-10上进行剪枝测试

echo "开始 Wanda-Diff 测试..."

# 设置参数
MODEL_PATH="google/ddpm-cifar10-32"
SAVE_PATH="run/test_wanda_diff"
PRUNING_RATIO=0.3
BATCH_SIZE=32
DEVICE="cuda:0"

# 创建输出目录
mkdir -p $SAVE_PATH

# 运行Wanda-Diff剪枝
python ddpm_prune.py \
    --model_path $MODEL_PATH \
    --save_path $SAVE_PATH \
    --pruning_ratio $PRUNING_RATIO \
    --batch_size $BATCH_SIZE \
    --device $DEVICE \
    --pruner wanda-diff \
    --dataset cifar10 \
    --wanda_calib_steps 512 \
    --wanda_time_strategy mean \
    --wanda_target_steps all \
    --wanda_activation_strategy mean \
    --wanda_analyze_activations \
    --wanda_save_analysis "$SAVE_PATH/activation_analysis.png"

echo "Wanda-Diff测试完成！"
echo "结果保存在: $SAVE_PATH"
echo "激活分析图保存在: $SAVE_PATH/activation_analysis.png" 