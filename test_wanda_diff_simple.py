#!/usr/bin/env python3
"""
简单的Wanda-Diff测试脚本
快速验证实现是否正常工作
"""

import torch
import os
import sys
from diffusers import DDPMPipeline
import utils
from utils.pruners import prune_wanda_diff

def main():
    print("=" * 60)
    print("Wanda-Diff 简单测试")
    print("=" * 60)
    
    # 设置参数
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    model_path = "google/ddpm-cifar10-32"
    save_path = "run/simple_test"
    pruning_ratio = 0.2  # 20% 剪枝
    batch_size = 16  # 较小的批次
    num_calib_steps = 256  # 较少的校准步数
    
    print(f"设备: {device}")
    print(f"模型: {model_path}")
    print(f"剪枝比率: {pruning_ratio:.1%}")
    print(f"校准步数: {num_calib_steps}")
    print("-" * 60)
    
    try:
        # 创建输出目录
        os.makedirs(save_path, exist_ok=True)
        
        # 加载模型
        print("1. 加载预训练模型...")
        pipeline = DDPMPipeline.from_pretrained(model_path).to(device)
        print("   ✓ 模型加载成功")
        
        # 准备数据
        print("2. 准备校准数据...")
        dataset = utils.get_calibration_dataset("cifar10", num_samples=512)
        dataloader = torch.utils.data.DataLoader(
            dataset, batch_size=batch_size, shuffle=True, num_workers=2
        )
        print("   ✓ 数据准备完成")
        
        # 执行剪枝
        print("3. 执行 Wanda-Diff 剪枝...")
        pipeline = prune_wanda_diff(
            pipeline=pipeline,
            train_dataloader=dataloader,
            pruning_ratio=pruning_ratio,
            device=device,
            num_calib_steps=num_calib_steps,
            time_strategy='mean',
            target_steps='all',
            activation_strategy='mean',
            analyze_activations=True,
            save_analysis=os.path.join(save_path, "analysis.png"),
            verbose=True
        )
        print("   ✓ 剪枝完成")
        
        # 保存模型
        print("4. 保存剪枝后的模型...")
        pipeline.save_pretrained(save_path)
        print("   ✓ 模型保存完成")
        
        # 生成测试图像
        print("5. 生成测试图像...")
        with torch.no_grad():
            generator = torch.Generator(device=device).manual_seed(42)
            images = pipeline(
                num_inference_steps=50, 
                batch_size=4, 
                generator=generator, 
                output_type="numpy"
            ).images
            
            # 保存图像
            import torchvision
            images_tensor = torch.from_numpy(images).permute([0, 3, 1, 2])
            os.makedirs(os.path.join(save_path, 'vis'), exist_ok=True)
            torchvision.utils.save_image(
                images_tensor, 
                os.path.join(save_path, 'vis', 'test_samples.png'),
                nrow=2
            )
        print("   ✓ 测试图像生成完成")
        
        print("=" * 60)
        print("✓ Wanda-Diff 测试成功完成！")
        print(f"✓ 结果保存在: {save_path}")
        print(f"✓ 测试图像: {save_path}/vis/test_samples.png")
        print(f"✓ 激活分析: {save_path}/analysis.png")
        print("=" * 60)
        
        return True
        
    except Exception as e:
        print(f"✗ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 