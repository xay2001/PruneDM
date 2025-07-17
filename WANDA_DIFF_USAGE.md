# Wanda-Diff 使用指南

Wanda-Diff 是将 Wanda 剪枝算法应用于扩散模型（DDPM）的实现。本指南将帮助您快速上手使用 Wanda-Diff 进行模型压缩。

## 快速开始

### 1. 环境准备

确保您已经安装了所有依赖：

```bash
pip install -r requirements.txt
```

### 2. 基本使用

#### 使用预训练模型进行剪枝

```bash
python ddpm_prune.py \
    --model_path google/ddpm-cifar10-32 \
    --save_path run/wanda_diff_test \
    --pruning_ratio 0.3 \
    --pruner wanda-diff \
    --device cuda:0 \
    --batch_size 32 \
    --dataset cifar10
```

#### 带激活分析的完整测试

```bash
python ddpm_prune.py \
    --model_path google/ddpm-cifar10-32 \
    --save_path run/wanda_diff_complete \
    --pruning_ratio 0.3 \
    --pruner wanda-diff \
    --device cuda:0 \
    --batch_size 32 \
    --dataset cifar10 \
    --wanda_analyze_activations \
    --wanda_save_analysis "run/wanda_diff_complete/analysis.png" \
    --wanda_calib_steps 1024 \
    --wanda_time_strategy mean \
    --wanda_target_steps all
```

### 3. 参数说明

#### 基本参数
- `--model_path`: 预训练模型路径
- `--save_path`: 剪枝后模型保存路径
- `--pruning_ratio`: 剪枝比率 (0.0-1.0)
- `--pruner`: 剪枝方法，选择 `wanda-diff`
- `--device`: 计算设备
- `--batch_size`: 批次大小
- `--dataset`: 数据集名称

#### Wanda-Diff 特定参数
- `--wanda_calib_steps`: 校准步数 (默认: 1024)
- `--wanda_time_strategy`: 时间步聚合策略 (mean/max/median/weighted_mean)
- `--wanda_target_steps`: 目标时间步范围 (all/early/late/middle/start-end)
- `--wanda_activation_strategy`: 激活聚合策略 (mean/max/median)
- `--wanda_analyze_activations`: 是否进行激活分析
- `--wanda_save_analysis`: 激活分析图保存路径

## 高级用法

### 1. 激活分析实验

在应用剪枝之前，您可以先分析模型的激活分布，验证 Wanda 方法论的适用性：

```bash
python experiments/analyze_activations.py \
    --model_path google/ddpm-cifar10-32 \
    --dataset cifar10 \
    --device cuda:0 \
    --num_samples 1024 \
    --save_path ./activation_analysis_results
```

### 2. 不同时间步策略

#### 关注早期去噪阶段（高噪声）
```bash
python ddpm_prune.py \
    --model_path google/ddpm-cifar10-32 \
    --save_path run/wanda_diff_early \
    --pruning_ratio 0.3 \
    --pruner wanda-diff \
    --wanda_target_steps early
```

#### 关注晚期去噪阶段（低噪声）
```bash
python ddpm_prune.py \
    --model_path google/ddpm-cifar10-32 \
    --save_path run/wanda_diff_late \
    --pruning_ratio 0.3 \
    --pruner wanda-diff \
    --wanda_target_steps late
```

#### 自定义时间步范围
```bash
python ddpm_prune.py \
    --model_path google/ddpm-cifar10-32 \
    --save_path run/wanda_diff_custom \
    --pruning_ratio 0.3 \
    --pruner wanda-diff \
    --wanda_target_steps "100-500"
```

### 3. 不同聚合策略

#### 使用加权平均（给后期时间步更高权重）
```bash
python ddpm_prune.py \
    --model_path google/ddpm-cifar10-32 \
    --save_path run/wanda_diff_weighted \
    --pruning_ratio 0.3 \
    --pruner wanda-diff \
    --wanda_time_strategy weighted_mean
```

#### 使用最大值聚合
```bash
python ddpm_prune.py \
    --model_path google/ddpm-cifar10-32 \
    --save_path run/wanda_diff_max \
    --pruning_ratio 0.3 \
    --pruner wanda-diff \
    --wanda_time_strategy max
```

## 脚本化使用

您也可以使用提供的脚本：

```bash
bash scripts/test_wanda_diff.sh
```

## 输出文件说明

剪枝完成后，在保存路径中您会找到：

- `config.json`: 模型配置文件
- `diffusion_pytorch_model.safetensors`: 剪枝后的模型权重
- `scheduler/`: 调度器配置
- `vis/after_pruning.png`: 剪枝后生成的样本图像
- `wanda_analysis.png`: 激活分析图（如果启用）

## 性能评估

### 1. 生成样本

剪枝后的模型会自动生成一些样本图像用于快速评估。

### 2. FID 评分

您可以使用项目中的 FID 评分脚本进行定量评估：

```bash
# 预计算数据集统计
python fid_score.py --save-stats data/cifar10_images run/fid_stats_cifar10.npz --device cuda:0

# 计算剪枝模型的 FID 分数
python fid_score.py run/wanda_diff_test/vis run/fid_stats_cifar10.npz --device cuda:0
```

## 常见问题

### Q: 如何选择合适的剪枝比率？
A: 建议从较小的剪枝比率开始（如 0.1-0.3），逐步增加。一般来说：
- 10-30%: 性能损失很小
- 30-50%: 中等性能损失
- 50%+: 可能有显著性能损失

### Q: 什么时候 Wanda-Diff 效果最好？
A: 当激活分析显示存在显著的"涌现大幅值特征"时（离群值比率 > 5%）。

### Q: 如何处理内存不足的问题？
A: 可以：
- 减小 `--batch_size`
- 减少 `--wanda_calib_steps`
- 使用更小的数据集

### Q: 剪枝后模型性能下降很多怎么办？
A: 可以尝试：
- 降低剪枝比率
- 使用不同的时间步策略
- 进行后续微调（虽然 Wanda 的目标是免微调）

## 实验建议

1. **首先运行激活分析**：验证 Wanda 方法论的适用性
2. **从小剪枝比率开始**：逐步测试 10%, 20%, 30% 等
3. **比较不同策略**：测试不同的时间步和聚合策略
4. **记录结果**：保存每次实验的 FID 分数和生成样本
5. **与基线对比**：与 magnitude pruning、random pruning 等方法对比

## 理论背景

Wanda-Diff 基于以下关键思想：

1. **权重-激活重要性度量**：`S = |W| × ||X||_2`
2. **结构化剪枝**：移除整个输出通道而非单个权重
3. **时间步感知**：考虑扩散模型的时间步依赖性
4. **一次性剪枝**：无需重训练或微调

更多理论细节请参考项目文档。 