CUDA_VISIBLE_DEVICES=1 python ddpm_prune.py \
--dataset cifar10 \
--model_path pretrained/ddpm_ema_cifar10 \
--save_path run/pruned/diff-pruning/ddpm_cifar10_pruned \
--pruning_ratio $1 \
--batch_size 128 \
--pruner diff-pruning \
--thr 0.05 \
--device cuda:1 \