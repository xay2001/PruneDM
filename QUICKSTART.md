# 🚀 快速开始指南

**5分钟快速体验两阶段混合剪枝**

## ⚡ 快速执行流程

### 🔧 环境准备（1分钟）

```bash
# 克隆并进入项目
git clone https://github.com/xay2001/PruneDM.git && cd PruneDM

# 创建并激活环境
conda create -n prunedm python=3.8 -y && conda activate prunedm

# 安装核心依赖
pip install torch torchvision diffusers accelerate swanlab
```

### 🎯 Stage 1: 结构化剪枝（2分钟）

```bash
# 运行magnitude剪枝
python ddpm_prune.py \
    --model_path "google/ddpm-cifar10-32" \
    --dataset "cifar10" \
    --pruning_method "magnitude" \
    --pruning_ratio 0.5 \
    --save_path "run/pruned/magnitude/ddpm_cifar10_pruned"
```

### 🎭 Stage 2: MaskPro训练（2分钟）

```bash
# 运行所有测试
python scripts/maskpro/test_foundation.py
python scripts/maskpro/test_integration.py
python scripts/maskpro/test_training.py

# 快速训练测试
./scripts/maskpro/run_maskpro_training.sh --quick_test
```

### 📊 模型评估（30秒）

```bash
# 快速评估
python scripts/maskpro/evaluation/evaluation_workflow.py --mode quick
```

## 🎉 预期结果

**测试通过标志:**
```
✅ Foundation Layer: 5/5 tests passed
✅ Integration Layer: 4/4 tests passed  
✅ Training Layer: 7/7 tests passed
✅ Quick evaluation completed
```

**评估指标:**
- 稀疏率: ~50%
- N:M合规性: >90%
- 样本质量: 可接受
- 性能提升: 可见

## 🔄 完整流程

如需完整训练和评估，请查看 [README.md](README.md) 获取详细步骤。

## ⚠️ 故障排除

**常见问题:**
```bash
# CUDA问题
conda install pytorch torchvision pytorch-cuda=11.8 -c pytorch -c nvidia

# 内存不足
./scripts/maskpro/run_maskpro_training.sh --device cpu --batch_size 4

# 测试失败
conda activate prunedm  # 确保环境正确
```

开始您的混合剪枝体验吧！ 🚀 