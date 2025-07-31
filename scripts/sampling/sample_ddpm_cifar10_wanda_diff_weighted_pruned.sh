CUDA_VISIBLE_DEVICES=1 python ddpm_sample.py \
--output_dir run/sample/wanda-diff-weighted/ddpm_cifar10_pruned \
--batch_size 128 \
--pruned_model_ckpt run/finetuned/wanda-diff-weighted/ddpm_cifar10_pruned_post_training/pruned/unet_ema_pruned.pth \
--model_path run/finetuned/wanda-diff-weighted/ddpm_cifar10_pruned_post_training \
--skip_type uniform 