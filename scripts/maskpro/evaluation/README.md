# MaskPro Evaluation Framework

This comprehensive evaluation framework provides multiple tools for assessing MaskPro trained models across different dimensions: model quality, sparsity effectiveness, hardware performance, and comparison with baseline methods.

## 🎯 Framework Overview

### Evaluation Modes

1. **Quick Evaluation** - Fast assessment during training
2. **Full Evaluation** - Comprehensive post-training analysis  
3. **Model Comparison** - Multi-model performance comparison
4. **Comprehensive Workflow** - Complete evaluation pipeline

### Key Metrics

- **Quality Metrics**: FID scores, sample generation quality
- **Sparsity Analysis**: Overall sparsity, N:M compliance rates
- **Performance Metrics**: Inference speed, memory usage
- **Compression Metrics**: Model size, compression ratios

## 📊 Evaluation Tools

### 1. Quick Evaluation (`quick_evaluate.py`)

Fast evaluation for training monitoring and rapid feedback.

```bash
# Basic quick evaluation
python scripts/maskpro/evaluation/quick_evaluate.py \
    --checkpoint run/maskpro/training/checkpoints/best_checkpoint.pt \
    --device cuda:0 \
    --num_samples 8

# Output: Basic sparsity stats, quick samples, performance check
```

**Use Cases:**
- Monitor training progress
- Quick model validation
- Development and debugging

### 2. Full Evaluation (`evaluate_maskpro_model.py`)

Comprehensive evaluation with detailed analysis and reporting.

```bash
# Full evaluation with baseline comparison
python scripts/maskpro/evaluation/evaluate_maskpro_model.py \
    --model_checkpoint run/maskpro/training/checkpoints/best_checkpoint.pt \
    --baseline_model run/pruned/magnitude/ddpm_cifar10_pruned \
    --device cuda:0 \
    --num_samples 64

# Skip expensive computations
python scripts/maskpro/evaluation/evaluate_maskpro_model.py \
    --model_checkpoint run/maskpro/training/checkpoints/best_checkpoint.pt \
    --skip_fid \
    --skip_benchmark
```

**Features:**
- Sample generation and quality assessment
- FID score computation
- Sparsity analysis with N:M compliance
- Performance benchmarking
- Memory usage analysis
- Visualization generation
- Comprehensive reporting

### 3. Model Comparison (`compare_models.py`)

Compare multiple pruning approaches side-by-side.

```bash
# Compare all available models
python scripts/maskpro/evaluation/compare_models.py \
    --original path/to/original/model \
    --magnitude run/pruned/magnitude/ddpm_cifar10_pruned \
    --maskpro run/maskpro/training/checkpoints/best_checkpoint.pt \
    --device cuda:0

# Compare with additional models
python scripts/maskpro/evaluation/compare_models.py \
    --magnitude run/pruned/magnitude/ddpm_cifar10_pruned \
    --maskpro run/maskpro/training/checkpoints/best_checkpoint.pt \
    --additional "CustomPruning:path/to/model:Custom pruning method"
```

**Outputs:**
- Comparison tables (CSV, Markdown)
- Visualization dashboards
- Performance analysis
- Compression efficiency plots

### 4. Evaluation Workflow (`evaluation_workflow.py`)

Orchestrates complete evaluation pipeline with intelligent workflow management.

```bash
# Comprehensive evaluation (recommended)
python scripts/maskpro/evaluation/evaluation_workflow.py \
    --mode comprehensive \
    --checkpoint run/maskpro/training/checkpoints/best_checkpoint.pt \
    --magnitude_model run/pruned/magnitude/ddpm_cifar10_pruned

# Auto-detect latest checkpoint
python scripts/maskpro/evaluation/evaluation_workflow.py \
    --mode comprehensive \
    --training_dir run/maskpro/training

# Individual modes
python scripts/maskpro/evaluation/evaluation_workflow.py --mode quick
python scripts/maskpro/evaluation/evaluation_workflow.py --mode full  
python scripts/maskpro/evaluation/evaluation_workflow.py --mode compare
```

## 📋 Evaluation Workflow Steps

### Step 1: Training Completion

After MaskPro training completes, you'll have:
- Checkpoints in `run/maskpro/training/checkpoints/`
- Training logs and metrics
- SwanLab experiment tracking data

### Step 2: Quick Validation

```bash
# Quick check of the trained model
python scripts/maskpro/evaluation/evaluation_workflow.py --mode quick
```

**Verifies:**
- Model loading works correctly
- Sparsity patterns are as expected
- Basic sample generation
- Performance is reasonable

### Step 3: Comprehensive Evaluation

```bash
# Full evaluation pipeline
python scripts/maskpro/evaluation/evaluation_workflow.py --mode comprehensive
```

**Generates:**
- Quality assessment (FID scores, samples)
- Detailed sparsity analysis
- Performance benchmarks
- Comparison with baselines
- Visual reports and dashboards

### Step 4: Analysis and Reporting

Review outputs in `run/maskpro/evaluation/`:

```
evaluation/
├── quick/                 # Quick evaluation results
│   └── quick_samples.png
├── full/                  # Comprehensive evaluation
│   ├── samples/          # Generated samples
│   ├── metrics/          # Detailed metrics
│   ├── analysis/         # Analysis files
│   ├── visualizations/   # Charts and plots
│   ├── evaluation_results.json
│   └── evaluation_summary.md
├── comparison/           # Model comparison
│   ├── samples/         # Samples from each model
│   ├── comparison_table.csv
│   ├── comparison_report.md
│   ├── comparison_dashboard.png
│   └── detailed_results.json
└── workflow_results.json # Overall workflow summary
```

## 🎯 Key Evaluation Metrics

### Model Quality
- **FID Score**: Fréchet Inception Distance vs CIFAR-10
- **Sample Quality**: Visual assessment of generated samples
- **Generation Consistency**: Stability across multiple runs

### Sparsity Effectiveness
- **Overall Sparsity**: Percentage of zero weights
- **N:M Compliance**: Adherence to N:M sparsity patterns
- **Layer-wise Analysis**: Sparsity distribution across layers
- **Hardware Compatibility**: Sparse Tensor Core utilization potential

### Performance Metrics
- **Inference Speed**: Forward pass timing (various batch sizes)
- **Memory Usage**: GPU memory allocation and peak usage
- **Throughput**: Samples generated per second
- **Scaling Behavior**: Performance across different workloads

### Compression Analysis
- **Model Size Reduction**: MB saved vs baseline
- **Parameter Reduction**: Total parameter count reduction
- **Compression Ratio**: Baseline size / compressed size
- **Efficiency**: Quality retention per compression unit

## 🔧 Advanced Usage

### Custom Evaluation Pipelines

Create custom evaluation scripts by importing framework components:

```python
from scripts.maskpro.evaluation.evaluate_maskpro_model import MaskProEvaluator

# Custom evaluation
evaluator = MaskProEvaluator(
    model_checkpoint="path/to/checkpoint.pt",
    baseline_model="path/to/baseline.pt",
    output_dir="custom/output"
)

# Run specific analyses
evaluator.load_models()
sparsity_results = evaluator.analyze_sparsity()
samples = evaluator.generate_samples(num_samples=32)
performance = evaluator.benchmark_inference_speed()
```

### Evaluation During Training

Integrate with training loop for continuous monitoring:

```python
# In training script
if epoch % validation_freq == 0:
    # Quick evaluation
    subprocess.run([
        "python", "scripts/maskpro/evaluation/quick_evaluate.py",
        "--checkpoint", f"checkpoints/checkpoint_epoch_{epoch}.pt",
        "--output_dir", f"eval/epoch_{epoch}"
    ])
```

### Batch Evaluation

Evaluate multiple checkpoints:

```bash
# Evaluate all checkpoints
for checkpoint in run/maskpro/training/checkpoints/checkpoint_epoch_*.pt; do
    python scripts/maskpro/evaluation/evaluation_workflow.py \
        --mode quick \
        --checkpoint "$checkpoint" \
        --output_dir "evaluation/$(basename $checkpoint .pt)"
done
```

## 📈 Interpreting Results

### Quality Assessment
- **FID < 50**: Good quality for CIFAR-10
- **FID < 20**: Excellent quality
- **Visual samples**: Should show diverse, realistic images

### Sparsity Analysis
- **Target sparsity achieved**: Verify pruning effectiveness
- **N:M compliance > 90%**: Good hardware compatibility
- **Layer-wise uniformity**: Balanced sparsity distribution

### Performance Gains
- **Speed improvement**: Actual vs theoretical speedup
- **Memory reduction**: Proportional to compression ratio
- **Scaling behavior**: Consistent across batch sizes

### Comparison Analysis
- **MaskPro vs Magnitude**: Quality retention with higher sparsity
- **Compression efficiency**: Best sparsity/quality trade-off
- **Hardware acceleration**: N:M sparsity advantages

## 🚀 Best Practices

### 1. Evaluation Timing
- **During training**: Use quick evaluation every few epochs
- **After training**: Run comprehensive evaluation on best checkpoint
- **Before deployment**: Full comparison with all baselines

### 2. Resource Management
- **GPU memory**: Monitor usage during benchmarking
- **Disk space**: Large sample sets can consume significant storage
- **Compute time**: Full evaluation can take 30-60 minutes

### 3. Metric Selection
- **Development**: Focus on sparsity compliance and basic quality
- **Research**: Include FID scores and detailed analysis
- **Production**: Emphasize performance and memory metrics

### 4. Baseline Comparison
- Always compare against magnitude-only pruning
- Include original model when available
- Document hardware-specific performance gains

## 🛠️ Troubleshooting

### Common Issues

**Model Loading Failures**
```bash
# Check checkpoint format
python -c "import torch; print(torch.load('checkpoint.pt', map_location='cpu').keys())"

# Verify model architecture compatibility
python scripts/maskpro/evaluation/quick_evaluate.py --checkpoint checkpoint.pt
```

**Memory Issues**
```bash
# Reduce sample count
python evaluation_workflow.py --mode full --num_samples 32

# Skip memory-intensive computations
python evaluation_workflow.py --mode full --skip_fid --skip_benchmark
```

**Performance Issues**
```bash
# Use CPU for testing
python evaluation_workflow.py --device cpu

# Skip benchmarking on slow systems
python evaluation_workflow.py --skip_benchmark
```

### Getting Help

1. Check evaluation logs in output directories
2. Verify model checkpoints are valid
3. Ensure sufficient GPU memory
4. Review SwanLab training logs for anomalies

## 📚 References

- **N:M Sparsity**: [NVIDIA Sparse Tensor Cores](https://developer.nvidia.com/blog/accelerating-inference-with-sparsity-using-ampere-and-tensorrt/)
- **FID Scores**: [Fréchet Inception Distance](https://arxiv.org/abs/1706.08500)
- **MaskPro Algorithm**: [Original MaskPro Paper](https://github.com/woodenchild95/Maskpro)
- **Diffusion Models**: [DDPM](https://arxiv.org/abs/2006.11239)

For more information, see the main project documentation and SwanLab experiment tracking. 