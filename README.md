# 🎭 Diffusion MaskPro: 两阶段混合剪枝系统

**基于MaskPro的扩散模型混合剪枝框架 - 结构化剪枝 + N:M稀疏学习**

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![CUDA](https://img.shields.io/badge/CUDA-11.0+-green.svg)](https://developer.nvidia.com/cuda-toolkit)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

## 🎯 项目概述

本项目实现了一个创新的两阶段混合剪枝系统，将 [Diff-Pruning](https://github.com/VainF/Diff-Pruning) 的结构化剪枝与 [MaskPro](https://github.com/woodenchild95/Maskpro) 的N:M稀疏学习相结合，用于压缩扩散模型。

### 🏗️ 核心架构

```
两阶段混合剪枝流程 (Prune-then-Learn)
┌─────────────────────────────────────────────────────────────────┐
│                    Stage 1: 结构化剪枝                           │
│  ┌─────────────┐    ┌──────────────┐    ┌─────────────────┐    │
│  │ 原始DDPM    │───▶│ Magnitude    │───▶│ 结构化剪枝模型    │    │
│  │ 模型        │    │ 剪枝         │    │                 │    │
│  └─────────────┘    └──────────────┘    └─────────────────┘    │
└─────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│                    Stage 2: N:M稀疏学习                          │
│  ┌─────────────────┐    ┌──────────────┐    ┌─────────────────┐│
│  │ 结构化剪枝模型    │───▶│ MaskPro      │───▶│ 混合剪枝模型    ││
│  │                 │    │ N:M学习      │    │ (结构化+N:M)   ││
│  └─────────────────┘    └──────────────┘    └─────────────────┘│
└─────────────────────────────────────────────────────────────────┘
```

### ✨ 主要特性

- **🎯 两阶段设计**: 结构化剪枝 + N:M稀疏学习
- **⚡ 硬件友好**: 支持NVIDIA Sparse Tensor Core加速
- **📊 智能学习**: 基于策略梯度的mask学习算法
- **🔬 完整评估**: 包含质量、性能、压缩率等多维度评估
- **📈 实验追踪**: 集成SwanLab进行训练监控
- **🛠️ 模块化设计**: 易于扩展和定制

## 🚀 快速开始

### 前置要求

- **Python**: 3.8+
- **PyTorch**: 2.0+
- **CUDA**: 11.0+ (推荐用于GPU加速)
- **内存**: 至少16GB RAM
- **存储**: 至少10GB可用空间

### 环境设置

```bash
# 1. 克隆项目
git clone https://github.com/xay2001/PruneDM.git
cd PruneDM

# 2. 创建conda环境
conda create -n prunedm python=3.8
conda activate prunedm

# 3. 安装依赖
pip install torch torchvision diffusers accelerate
pip install numpy matplotlib seaborn pandas tqdm
pip install swanlab  # 可选：用于实验追踪

# 4. 安装额外依赖
pip install torchmetrics  # 用于FID计算
pip install transformers  # diffusers依赖
```

## 📋 完整执行流程

### 🔄 总体流程概览

```
1. 环境准备 ✅
2. Stage 1: 结构化剪枝 ⏳
3. Sprint 1: Foundation Layer测试 ⏳  
4. Sprint 2: Integration Layer测试 ⏳
5. Stage 2: MaskPro训练 ⏳
6. Sprint 3: Training Layer测试 ⏳
7. 模型评估 ⏳
8. 结果分析 ⏳
```

---

## 🎯 Stage 1: 结构化剪枝

### 步骤1: 运行Magnitude剪枝

```bash
# 激活环境
conda activate prunedm

# 执行magnitude剪枝
python ddpm_prune.py \
    --model_path "google/ddpm-cifar10-32" \
    --dataset "cifar10" \
    --pruning_method "magnitude" \
    --pruning_ratio 0.5 \
    --save_path "run/pruned/magnitude/ddpm_cifar10_pruned"
```

**预期输出:**
- 剪枝后的模型保存在 `run/pruned/magnitude/ddpm_cifar10_pruned/`
- 包含 `pruned/unet_pruned.pth` 文件
- 显示剪枝统计信息

### 步骤2: 验证Stage 1结果

```bash
# 检查剪枝结果
ls -la run/pruned/magnitude/ddpm_cifar10_pruned/
ls -la run/pruned/magnitude/ddpm_cifar10_pruned/pruned/

# 生成样本验证质量
bash scripts/sampling/sample_ddpm_cifar10_magnitude_pruned.sh
```

---

## 🧪 Sprint 1: Foundation Layer测试

### 步骤3: 测试MaskPro核心组件

```bash
# 运行Foundation Layer测试
python scripts/maskpro/test_foundation.py
```

**测试内容:**
- ✅ MaskProLayer功能测试
- ✅ N:M兼容性验证  
- ✅ 模型包装测试
- ✅ 参数分离验证
- ✅ 训练模拟测试

**预期结果:**
```
🚀 Foundation Layer Test Suite
==========================================
✅ MaskProLayer Creation PASSED
✅ N:M Compatibility Check PASSED  
✅ Model Wrapping PASSED
✅ Parameter Separation PASSED
✅ Training Simulation PASSED
📊 Foundation Test Results: 5/5 tests passed
🎉 Foundation Layer is ready for integration!
```

---

## 🔗 Sprint 2: Integration Layer测试

### 步骤4: 提取初始Mask

```bash
# 从magnitude剪枝模型提取N:M mask
python scripts/maskpro/extract_initial_masks.py \
    --pruned_model_path "run/pruned/magnitude/ddpm_cifar10_pruned" \
    --output_dir "run/maskpro/initial_masks" \
    --n 2 --m 4
```

**预期输出:**
- 提取到的mask保存在 `run/maskpro/initial_masks/`
- 显示提取统计和N:M兼容性分析

### 步骤5: 计算基准损失

```bash
# 计算基准扩散损失
python scripts/maskpro/diffusion_inference_loss.py \
    --pruned_model_path "run/pruned/magnitude/ddpm_cifar10_pruned" \
    --initial_masks_dir "run/maskpro/initial_masks" \
    --output_file "run/maskpro/baseline_loss.json"
```

### 步骤6: 运行Integration测试

```bash
# 运行Integration Layer测试
python scripts/maskpro/test_integration.py
```

**测试内容:**
- ✅ Mask提取测试
- ✅ 扩散模型兼容性测试
- ✅ N:M模式分析测试
- ✅ MaskPro训练设置测试

**预期结果:**
```
🚀 Sprint 2 Integration Test Suite
========================================
✅ Mask Extraction PASSED
✅ Diffusion Model Compatibility PASSED
✅ N:M Pattern Analysis PASSED
✅ MaskPro Training Setup PASSED
📊 Integration Test Results: 4/4 tests passed
🎉 Integration Layer is ready for training!
```

---

## 🎭 Stage 2: MaskPro训练

### 步骤7: 配置训练参数

编辑训练配置文件 `scripts/maskpro/configs/diffusion_maskpro_config.yaml`:

```yaml
# 关键配置
model:
  pruned_model_path: "run/pruned/magnitude/ddpm_cifar10_pruned"
  n: 2  # N:M中的N
  m: 4  # N:M中的M

training:
  epochs: 50
  batch_size: 16
  model_lr: 1e-5
  mask_lr: 1e-3

logging:
  use_swanlab: true
  experiment_name: "ddpm-cifar10-magnitude-2-4"
```

### 步骤8: 运行MaskPro训练

```bash
# 方式1: 使用便捷脚本（推荐）
./scripts/maskpro/run_maskpro_training.sh

# 方式2: 快速测试模式
./scripts/maskpro/run_maskpro_training.sh --quick_test

# 方式3: 直接调用训练脚本
python scripts/maskpro/diffusion_maskpro_train.py \
    --config scripts/maskpro/configs/diffusion_maskpro_config.yaml
```

**训练过程监控:**
- 📈 SwanLab仪表板: 实时训练指标
- 📁 检查点: `run/maskpro/training/checkpoints/`
- 📊 日志: `run/maskpro/training/logs/`

**预期训练输出:**
```
🎭 Diffusion MaskPro Training
📋 DIFFUSION MASKPRO TRAINING CONFIGURATION
==========================================
🎯 Model: run/pruned/magnitude/ddpm_cifar10_pruned
   N:M Pattern: 2:4
🚀 Training:
   Epochs: 50
   Batch Size: 16
   Model LR: 1e-05
   Mask LR: 0.001
⏰ Training started at: 2024-XX-XX XX:XX:XX
```

---

## 🧪 Sprint 3: Training Layer测试

### 步骤9: 验证训练流程

```bash
# 运行Training Layer测试
python scripts/maskpro/test_training.py
```

**测试内容:**
- ✅ 配置加载测试
- ✅ 模型加载测试
- ✅ 数据加载测试
- ✅ 训练步骤测试
- ✅ 验证循环测试
- ✅ 检查点保存测试
- ✅ 小型训练运行测试

**预期结果:**
```
🚀 Sprint 3 Training Test Suite
=====================================
✅ Configuration Loading PASSED
✅ Model Loading PASSED
✅ Data Loading PASSED
✅ Training Step PASSED
✅ Validation PASSED
✅ Checkpoint Saving PASSED
✅ Mini Training Run PASSED
📊 Training Test Results: 7/7 tests passed
🎉 All training tests passed! Sprint 3 is ready!
```

---

## 📊 模型评估

### 步骤10: 运行完整评估

```bash
# 方式1: 综合评估（推荐）
python scripts/maskpro/evaluation/evaluation_workflow.py \
    --mode comprehensive \
    --checkpoint run/maskpro/training/checkpoints/best_checkpoint.pt \
    --magnitude_model run/pruned/magnitude/ddpm_cifar10_pruned

# 方式2: 快速评估
python scripts/maskpro/evaluation/evaluation_workflow.py --mode quick

# 方式3: 仅模型比较
python scripts/maskpro/evaluation/evaluation_workflow.py --mode compare
```

### 评估输出结构

```
run/maskpro/evaluation/
├── quick/                    # 快速评估结果
│   └── quick_samples.png    # 快速生成样本
├── full/                    # 完整评估结果
│   ├── samples/            # 生成样本
│   ├── metrics/            # 详细指标
│   ├── visualizations/     # 可视化图表
│   ├── evaluation_results.json
│   └── evaluation_summary.md
├── comparison/             # 模型比较结果
│   ├── comparison_table.csv
│   ├── comparison_report.md
│   ├── comparison_dashboard.png
│   └── detailed_results.json
└── workflow_results.json  # 工作流总结
```

### 关键评估指标

**模型质量:**
- 📊 FID Score (越低越好)
- 🎨 样本质量 (视觉评估)
- 🔄 生成一致性

**稀疏性效果:**
- 📉 总体稀疏率
- ✅ N:M合规性 (>90%为优)
- 📊 层级分布

**性能指标:**
- ⚡ 推理速度 (毫秒)
- 💾 内存使用 (GB)
- 🔥 吞吐量 (samples/s)

**压缩分析:**
- 📏 模型大小减少
- 🗜️ 压缩比率
- ⚖️ 质量保持率

---

## 📈 结果分析和验证

### 步骤11: 分析评估报告

1. **查看综合报告**
```bash
# 查看评估总结
cat run/maskpro/evaluation/full/evaluation_summary.md

# 查看模型比较
cat run/maskpro/evaluation/comparison/comparison_report.md
```

2. **检查关键指标**
```bash
# 查看详细结果
python -c "
import json
with open('run/maskpro/evaluation/workflow_results.json', 'r') as f:
    results = json.load(f)
    
print('📊 评估总结:')
print(f'时间戳: {results[\"timestamp\"]}')
print(f'检查点: {results[\"checkpoint\"]}')

for step, result in results['steps'].items():
    status = '✅' if result['status'] == 'success' else '❌'
    print(f'{status} {step}: {result[\"status\"]}')
"
```

3. **验证成功标准**

**质量标准:**
- ✅ FID Score < 50 (CIFAR-10)
- ✅ 样本质量可接受
- ✅ 生成稳定性良好

**稀疏性标准:**
- ✅ 达到目标稀疏率
- ✅ N:M合规性 > 90%
- ✅ 层级分布均匀

**性能标准:**
- ✅ 推理速度提升
- ✅ 内存占用减少
- ✅ 硬件兼容性良好

---

## 🔧 故障排除

### 常见问题及解决方案

#### 1. 环境相关问题

**问题: CUDA版本不匹配**
```bash
# 检查CUDA版本
nvidia-smi
python -c "import torch; print(torch.cuda.is_available(), torch.version.cuda)"

# 重新安装匹配的PyTorch
pip uninstall torch torchvision
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

**问题: 内存不足**
```bash
# 减少批次大小
./scripts/maskpro/run_maskpro_training.sh --batch_size 8

# 使用CPU训练
./scripts/maskpro/run_maskpro_training.sh --device cpu
```

#### 2. 训练相关问题

**问题: 训练收敛慢**
```yaml
# 调整配置文件
training:
  model_lr: 2e-5    # 增加模型学习率
  mask_lr: 2e-3     # 增加mask学习率
  gradient_clip_norm: 0.5  # 减少梯度裁剪
```

**问题: N:M合规性低**
```yaml
# 增加mask学习权重
training:
  mask_loss_weight: 2.0  # 从1.0增加到2.0
```

#### 3. 评估相关问题

**问题: FID计算失败**
```bash
# 跳过FID计算
python scripts/maskpro/evaluation/evaluation_workflow.py \
    --mode full --skip_fid
```

**问题: 模型加载失败**
```bash
# 使用快速评估检查
python scripts/maskpro/evaluation/quick_evaluate.py \
    --checkpoint run/maskpro/training/checkpoints/best_checkpoint.pt
```

### 日志分析

**训练日志位置:**
- SwanLab: 在线仪表板
- 本地日志: `run/maskpro/training/logs/`
- 检查点: `run/maskpro/training/checkpoints/`

**关键监控指标:**
- `main_loss`: 主要扩散损失
- `mask_loss`: mask学习损失  
- `sparsity_ratio`: 稀疏率
- `nm_compliance`: N:M合规性

---

## 📚 API文档和扩展

### 核心模块使用

```python
# 1. 使用MaskProLayer
from diffusion_maskpro import MaskProLayer

layer = MaskProLayer(
    original_layer=nn.Conv2d(64, 128, 3),
    n=2, m=4  # 2:4稀疏
)

# 2. 模型包装
from diffusion_maskpro import wrap_model_with_maskpro

wrapped_model = wrap_model_with_maskpro(
    model, 
    n=2, m=4,
    target_layers=[".*conv.*", ".*linear.*"]
)

# 3. 稀疏性分析
from diffusion_maskpro import get_model_sparsity_summary

summary = get_model_sparsity_summary(model)
print(f"总体稀疏率: {summary['overall_stats']['overall_sparsity']:.1%}")
```

### 自定义训练循环

```python
from diffusion_maskpro import DiffusionMaskProTrainer

# 自定义训练器
trainer = DiffusionMaskProTrainer("config.yaml")
trainer.load_model_and_scheduler()
trainer.apply_maskpro_wrappers()
trainer.setup_data_loaders()
trainer.setup_optimizers()

# 自定义训练步骤
for epoch in range(epochs):
    for batch in train_loader:
        metrics = trainer.training_step(batch)
        print(f"Loss: {metrics['total_loss']:.4f}")
```

### 自定义评估

```python
from scripts.maskpro.evaluation.evaluate_maskpro_model import MaskProEvaluator

evaluator = MaskProEvaluator(
    model_checkpoint="checkpoint.pt",
    baseline_model="baseline.pt"
)

# 运行特定评估
evaluator.load_models()
sparsity = evaluator.analyze_sparsity()
samples = evaluator.generate_samples(32)
performance = evaluator.benchmark_inference_speed()
```

---

## 🎯 后续步骤和优化

### 1. 模型部署

```bash
# 转换为推理优化格式
python scripts/deployment/optimize_for_inference.py \
    --checkpoint run/maskpro/training/checkpoints/best_checkpoint.pt \
    --output_dir deployment/optimized

# 硬件特定优化
python scripts/deployment/tensorrt_optimization.py \
    --model deployment/optimized \
    --precision fp16
```

### 2. 实验扩展

**不同N:M模式:**
```bash
# 尝试1:4稀疏
./scripts/maskpro/run_maskpro_training.sh \
    --experiment_name "ddpm-1-4-sparsity"
# 修改config中的n=1, m=4

# 尝试4:8稀疏  
./scripts/maskpro/run_maskpro_training.sh \
    --experiment_name "ddpm-4-8-sparsity"
# 修改config中的n=4, m=8
```

**不同剪枝方法:**
```bash
# Taylor剪枝 + MaskPro
python ddpm_prune.py --pruning_method taylor
# 然后运行MaskPro训练

# Random剪枝 + MaskPro  
python ddpm_prune.py --pruning_method random
# 然后运行MaskPro训练
```

### 3. 大规模实验

```bash
# 批量实验脚本
for method in magnitude taylor random; do
    for pattern in "2:4" "1:4" "4:8"; do
        echo "Running $method + $pattern"
        # 运行剪枝
        # 运行MaskPro
        # 运行评估
    done
done
```

### 4. 论文和报告

- 📊 实验结果整理
- 📈 性能对比分析
- 🎯 硬件加速验证
- 📝 方法论总结

---

## 📖 参考资料

### 核心论文
- **DDPM**: [Denoising Diffusion Probabilistic Models](https://arxiv.org/abs/2006.11239)
- **Diff-Pruning**: [Diff-Pruning: Structural Pruning for Diffusion Models](https://github.com/VainF/Diff-Pruning)
- **MaskPro**: [MaskPro: N:M Sparsity Learning Framework](https://github.com/woodenchild95/Maskpro)
- **N:M Sparsity**: [Accelerating Sparse Deep Neural Networks](https://arxiv.org/abs/2104.08378)

### 技术文档
- **NVIDIA Sparse Tensor Cores**: [官方文档](https://developer.nvidia.com/blog/accelerating-inference-with-sparsity-using-ampere-and-tensorrt/)
- **PyTorch Pruning**: [官方教程](https://pytorch.org/tutorials/intermediate/pruning_tutorial.html)
- **Diffusers Library**: [文档](https://huggingface.co/docs/diffusers/)

### 相关项目
- **Torch-Pruning**: [结构化剪枝库](https://github.com/VainF/Torch-Pruning)
- **SwanLab**: [实验追踪平台](https://swanlab.cn/)

---

## 🤝 贡献和支持

### 贡献指南
1. Fork项目
2. 创建特性分支
3. 提交更改
4. 创建Pull Request

### 问题报告
- 🐛 Bug报告: 使用GitHub Issues
- 💡 功能请求: 使用GitHub Discussions  
- ❓ 使用问题: 查看文档或提Issue

### 开发计划
- [ ] 支持更多扩散模型架构
- [ ] 集成更多剪枝方法
- [ ] 硬件特定优化
- [ ] 自动化超参数搜索

---

## 📄 许可证

本项目采用MIT许可证 - 详见 [LICENSE](LICENSE) 文件。

---

## 🎉 总结

这个两阶段混合剪枝系统成功地将结构化剪枝与N:M稀疏学习相结合，为扩散模型压缩提供了一个完整的解决方案。通过本文档的详细步骤，您可以：

1. ✅ 完成环境设置和依赖安装
2. ✅ 执行完整的两阶段剪枝流程
3. ✅ 运行全面的测试验证
4. ✅ 进行多维度模型评估
5. ✅ 分析结果并优化模型

**关键优势:**
- 🎯 **高压缩率**: 结构化+N:M双重压缩
- ⚡ **硬件友好**: 支持现代GPU加速
- 🔬 **质量保持**: 智能学习保持模型性能
- 📊 **完整评估**: 多维度性能分析

**预期效果:**
- 模型大小减少50-80%
- 推理速度提升20-40%  
- 质量损失<5%
- 硬件兼容性优秀

开始您的混合剪枝之旅吧！🚀
