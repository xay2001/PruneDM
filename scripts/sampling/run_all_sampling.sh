#!/bin/bash

# 在GPU 0上运行magnitude采样
CUDA_VISIBLE_DEVICES=0 bash sample_ddpm_cifar10_magnitude_pruned.sh &

# 在GPU 1上运行diff采样
CUDA_VISIBLE_DEVICES=1 bash sample_ddpm_cifar10_diff_pruned.sh &

# 在GPU 2上运行random采样
CUDA_VISIBLE_DEVICES=2 bash sample_ddpm_cifar10_random_pruned.sh &

# 在GPU 3上运行taylor采样
CUDA_VISIBLE_DEVICES=3 bash sample_ddpm_cifar10_taylor_pruned.sh &

# 等待所有任务完成
wait

echo "所有采样任务已完成！" 