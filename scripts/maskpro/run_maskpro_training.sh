#!/bin/bash
# Diffusion MaskPro Training Script
# Two-stage hybrid pruning: Magnitude + N:M sparsity learning

set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Print colored output
print_color() {
    printf "${1}${2}${NC}\n"
}

print_color $BLUE "🎭 Diffusion MaskPro Training Launcher"
print_color $BLUE "======================================="

# Check if running in conda environment
if [[ "$CONDA_DEFAULT_ENV" != "prunedm" ]]; then
    print_color $YELLOW "⚠️  Warning: Not in 'prunedm' conda environment"
    print_color $YELLOW "   Current environment: $CONDA_DEFAULT_ENV"
    echo "   Activate with: conda activate prunedm"
    read -p "   Continue anyway? [y/N]: " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

# Configuration
CONFIG_FILE="scripts/maskpro/configs/diffusion_maskpro_config.yaml"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

# Parse command line arguments
QUICK_TEST=false
DEVICE=""
BATCH_SIZE=""
EPOCHS=""
EXPERIMENT_NAME=""
NO_SWANLAB=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --quick_test)
            QUICK_TEST=true
            shift
            ;;
        --device)
            DEVICE="$2"
            shift 2
            ;;
        --batch_size)
            BATCH_SIZE="$2"
            shift 2
            ;;
        --epochs)
            EPOCHS="$2"
            shift 2
            ;;
        --experiment_name)
            EXPERIMENT_NAME="$2"
            shift 2
            ;;
        --no_swanlab)
            NO_SWANLAB=true
            shift
            ;;
        --help)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --quick_test          Run in quick test mode (small dataset, few epochs)"
            echo "  --device DEVICE       Override device (e.g., cuda:0, cpu)"
            echo "  --batch_size SIZE     Override batch size"
            echo "  --epochs NUM          Override number of epochs"
            echo "  --experiment_name NAME Override experiment name"
            echo "  --no_swanlab          Disable SwanLab logging"
            echo "  --help                Show this help message"
            exit 0
            ;;
        *)
            print_color $RED "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Change to project root
cd "$PROJECT_ROOT"

print_color $GREEN "🔍 Pre-flight checks..."

# Check if configuration file exists
if [[ ! -f "$CONFIG_FILE" ]]; then
    print_color $RED "❌ Configuration file not found: $CONFIG_FILE"
    exit 1
fi
print_color $GREEN "✓ Configuration file found"

# Check if pruned model exists
PRUNED_MODEL_PATH="run/pruned/magnitude/ddpm_cifar10_pruned"
if [[ ! -d "$PRUNED_MODEL_PATH" ]]; then
    print_color $RED "❌ Pruned model directory not found: $PRUNED_MODEL_PATH"
    print_color $YELLOW "   You need to run magnitude pruning first with:"
    print_color $YELLOW "   python ddpm_prune.py --dataset cifar10 --save_path $PRUNED_MODEL_PATH"
    exit 1
fi
print_color $GREEN "✓ Pruned model directory found"

# Check for pruned model files
PRUNED_DIR="$PRUNED_MODEL_PATH/pruned"
if [[ -d "$PRUNED_DIR" ]]; then
    if ls "$PRUNED_DIR"/*.pth 1> /dev/null 2>&1; then
        print_color $GREEN "✓ Pruned model files found"
    else
        print_color $RED "❌ No .pth files found in $PRUNED_DIR"
        exit 1
    fi
else
    print_color $YELLOW "⚠️  No pruned subdirectory, will attempt pipeline loading"
fi

# Check if SwanLab is installed (if not disabled)
if [[ "$NO_SWANLAB" == false ]]; then
    if python -c "import swanlab" 2>/dev/null; then
        print_color $GREEN "✓ SwanLab available"
    else
        print_color $YELLOW "⚠️  SwanLab not installed, will disable logging"
        NO_SWANLAB=true
    fi
fi

# Check GPU availability if not specified
if [[ -z "$DEVICE" ]]; then
    if python -c "import torch; print('CUDA available:', torch.cuda.is_available())" | grep -q "True"; then
        print_color $GREEN "✓ CUDA available, will use GPU"
        DEVICE="cuda:0"
    else
        print_color $YELLOW "⚠️  CUDA not available, will use CPU"
        DEVICE="cpu"
    fi
fi

print_color $GREEN "✅ All checks passed!"

# Run tests first if in quick test mode
if [[ "$QUICK_TEST" == true ]]; then
    print_color $BLUE "\n🧪 Running Sprint 3 tests first..."
    if python scripts/maskpro/test_training.py; then
        print_color $GREEN "✓ Tests passed, proceeding with training"
    else
        print_color $RED "❌ Tests failed, aborting training"
        exit 1
    fi
fi

# Prepare command line arguments
ARGS="--config $CONFIG_FILE"

if [[ -n "$DEVICE" ]]; then
    ARGS="$ARGS --device $DEVICE"
fi

if [[ -n "$BATCH_SIZE" ]]; then
    ARGS="$ARGS --batch_size $BATCH_SIZE"
fi

if [[ -n "$EPOCHS" ]]; then
    ARGS="$ARGS --epochs $EPOCHS"
fi

if [[ -n "$EXPERIMENT_NAME" ]]; then
    ARGS="$ARGS --experiment_name $EXPERIMENT_NAME"
fi

if [[ "$NO_SWANLAB" == true ]]; then
    ARGS="$ARGS --no_swanlab"
fi

if [[ "$QUICK_TEST" == true ]]; then
    ARGS="$ARGS --quick_test"
fi

# Display training configuration
print_color $BLUE "\n📋 Training Configuration:"
print_color $BLUE "=========================="
print_color $BLUE "Config file: $CONFIG_FILE"
print_color $BLUE "Device: $DEVICE"
if [[ -n "$BATCH_SIZE" ]]; then
    print_color $BLUE "Batch size: $BATCH_SIZE"
fi
if [[ -n "$EPOCHS" ]]; then
    print_color $BLUE "Epochs: $EPOCHS"
fi
if [[ "$QUICK_TEST" == true ]]; then
    print_color $YELLOW "Mode: Quick Test (small dataset, few epochs)"
else
    print_color $BLUE "Mode: Full Training"
fi
if [[ "$NO_SWANLAB" == true ]]; then
    print_color $YELLOW "Logging: Disabled"
else
    print_color $BLUE "Logging: SwanLab enabled"
fi

# Start training
print_color $GREEN "\n🚀 Starting MaskPro training..."
print_color $GREEN "Command: python scripts/maskpro/diffusion_maskpro_train.py $ARGS"
print_color $GREEN "================================="

# Execute training
python scripts/maskpro/diffusion_maskpro_train.py $ARGS

# Check exit code
if [[ $? -eq 0 ]]; then
    print_color $GREEN "\n🎉 Training completed successfully!"
    
    # Show output directories
    if [[ -d "run/maskpro/training" ]]; then
        print_color $GREEN "\n📁 Output files created in:"
        print_color $GREEN "   Checkpoints: run/maskpro/training/checkpoints/"
        print_color $GREEN "   Logs: run/maskpro/training/logs/"
        print_color $GREEN "   Samples: run/maskpro/training/samples/"
        print_color $GREEN "   Analysis: run/maskpro/training/analysis/"
    fi
    
    # Next steps
    print_color $BLUE "\n📋 Next Steps:"
    print_color $BLUE "1. Analyze training logs and metrics"
    print_color $BLUE "2. Generate samples with the trained model"
    print_color $BLUE "3. Evaluate sparsity and compression ratios"
    print_color $BLUE "4. Compare with magnitude-only pruned baseline"
    
else
    print_color $RED "\n❌ Training failed!"
    print_color $RED "Check the error messages above for details."
    exit 1
fi 