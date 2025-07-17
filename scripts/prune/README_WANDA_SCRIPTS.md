# Wanda-Diff 剪枝脚本使用指南

本目录包含了多种Wanda-Diff剪枝配置的脚本，用于在CIFAR-10数据集上对DDPM模型进行剪枝。

## 脚本概览

### 1. 标准配置 (`prune_ddpm_wanda_cifar10.sh`)
- **用途**: 标准的Wanda-Diff剪枝配置
- **特点**: 使用全部时间步，均值聚合策略
- **校准步数**: 1024
- **使用方法**: `bash scripts/prune/prune_ddpm_wanda_cifar10.sh 0.3`

### 2. 早期时间步配置 (`prune_ddpm_wanda_early_steps_cifar10.sh`)
- **用途**: 专注于早期时间步的剪枝
- **特点**: 仅使用高噪声阶段的激活信息
- **适用场景**: 当认为早期结构决定比细节重要时
- **使用方法**: `bash scripts/prune/prune_ddpm_wanda_early_steps_cifar10.sh 0.3`

### 3. 加权平均配置 (`prune_ddpm_wanda_weighted_cifar10.sh`)
- **用途**: 给后期时间步更高权重的剪枝
- **特点**: 使用加权平均策略，重视细节阶段
- **适用场景**: 想要保护图像细节质量时
- **使用方法**: `bash scripts/prune/prune_ddpm_wanda_weighted_cifar10.sh 0.3`

### 4. 快速测试配置 (`prune_ddpm_wanda_fast_cifar10.sh`)
- **用途**: 快速验证功能的测试配置
- **特点**: 较少校准步数(256)，中等时间步
- **适用场景**: 快速原型验证和调试
- **使用方法**: `bash scripts/prune/prune_ddpm_wanda_fast_cifar10.sh 0.3`

### 5. 配置对比脚本 (`compare_wanda_configurations.sh`)
- **用途**: 一次性运行所有配置进行对比
- **特点**: 系统性测试不同策略的效果
- **使用方法**: `bash scripts/prune/compare_wanda_configurations.sh 0.3`

## 参数说明

所有脚本都接受一个剪枝比率参数：
- `$1`: 剪枝比率 (0.0-1.0)
  - 例如: `0.3` 表示剪除30%的通道
  - 如果不提供，默认使用0.3

## 关键配置差异

| 脚本 | 时间步策略 | 目标时间步 | 校准步数 | 批大小 |
|------|-----------|-----------|---------|--------|
| 标准配置 | mean | all | 1024 | 64 |
| 早期时间步 | mean | early | 1024 | 64 |
| 加权平均 | weighted_mean | all | 1024 | 64 |
| 快速测试 | mean | middle | 256 | 32 |

## 输出目录结构

```
run/pruned/
├── wanda-diff/                    # 标准配置结果
├── wanda-diff-early/              # 早期时间步配置结果
├── wanda-diff-weighted/           # 加权平均配置结果
└── wanda-diff-fast/               # 快速测试配置结果
```

每个目录包含：
- 剪枝后的模型文件
- 激活分析图表 (`activation_analysis.png`)
- 剪枝日志和统计信息

## 使用建议

1. **首次使用**: 先运行快速测试配置验证环境
2. **性能对比**: 使用配置对比脚本系统性评估
3. **生产使用**: 根据需求选择标准配置或加权平均配置
4. **调试问题**: 检查激活分析图表理解剪枝决策

## 前置条件

确保已经：
- 安装所有依赖 (`pip install -r requirements.txt`)
- 下载预训练DDPM模型到 `pretrained/ddpm_ema_cifar10`
- 配置合适的CUDA设备 