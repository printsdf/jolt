#!/usr/bin/env python3
"""
超级显存优化的JOLT运行脚本
使用所有可能的显存优化技术
"""

import os
import sys
import torch
import torch.nn as nn
from accelerate import Accelerator
from accelerate.utils import set_seed
import traceback
import gc

def ultra_memory_optimization():
    """终极显存优化设置"""
    if torch.cuda.is_available():
        # 设置显存分配策略
        for i in range(torch.cuda.device_count()):
            torch.cuda.set_device(i)
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
        
        # 设置每个进程的显存限制
        torch.cuda.set_per_process_memory_fraction(0.8)  # 每个进程最多80%显存
        
        # 启用显存池
        os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True,max_split_size_mb:64'
        
        print("✓ 终极显存优化已启用")

def setup_ultra_distributed():
    """设置超级优化的分布式环境"""
    try:
        # 先进行显存优化
        ultra_memory_optimization()
        
        # 创建accelerator
        accelerator = Accelerator(
            mixed_precision='bf16',  # 使用bf16减少显存
            gradient_accumulation_steps=1,  # 不使用梯度累积
        )
        
        if accelerator.is_main_process:
            print(f"超级优化6GPU分布式设置:")
            print(f"  设备: {accelerator.device}")
            print(f"  进程数: {accelerator.num_processes}")
            print(f"  混合精度: {accelerator.mixed_precision}")
            
            # 显示显存状态
            if torch.cuda.is_available():
                total_free = 0
                for i in range(torch.cuda.device_count()):
                    props = torch.cuda.get_device_properties(i)
                    total = props.total_memory / 1024**3
                    allocated = torch.cuda.memory_allocated(i) / 1024**3
                    free = total - allocated
                    total_free += free
                    print(f"  GPU {i}: 总计 {total:.1f}GB, 可用 {free:.1f}GB")
                print(f"  总可用显存: {total_free:.1f}GB")
        
        return accelerator
        
    except Exception as e:
        print(f"超级分布式设置失败: {e}")
        traceback.print_exc()
        return None

def optimize_model_for_memory(model):
    """对模型进行显存优化"""
    try:
        # 启用梯度检查点（如果模型支持）
        if hasattr(model, 'gradient_checkpointing_enable'):
            model.gradient_checkpointing_enable()
            print("✓ 梯度检查点已启用")
        
        # 设置模型为eval模式以节省显存
        model.eval()
        
        # 禁用不必要的梯度计算
        for param in model.parameters():
            param.requires_grad = False
        
        print("✓ 模型显存优化完成")
        return model
        
    except Exception as e:
        print(f"模型优化失败: {e}")
        return model

def run_ultra_optimized_jolt():
    """运行超级优化的JOLT"""
    try:
        # 设置分布式环境
        accelerator = setup_ultra_distributed()
        if accelerator is None:
            return False
        
        # 导入模块
        from run_jolt import run_jolt
        from parse_args import parse_command_line
        from hf_api import get_model_and_tokenizer
        
        # 解析参数
        args = parse_command_line()
        set_seed(args.seed)
        
        if accelerator.is_main_process:
            print(f"超级优化参数:")
            print(f"  批次大小: {args.batch_size}")
            print(f"  训练样本: {args.train_size_limit}")
            print(f"  测试样本: {args.test_size_limit}")
            print(f"  最大生成长度: {args.max_generated_length}")
        
        # 加载模型（只在主进程显示进度）
        if accelerator.is_main_process:
            print("正在加载模型（显存优化模式）...")
        
        # 强制垃圾回收
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        model, tokenizer = get_model_and_tokenizer(args)
        
        # 优化模型显存使用
        model = optimize_model_for_memory(model)
        
        # 使用accelerator准备模型
        model = accelerator.prepare(model)
        
        if accelerator.is_main_process:
            print("模型加载完成，开始超级优化训练...")
            
            # 显示加载后的显存状态
            if torch.cuda.is_available():
                print("模型加载后显存状态:")
                for i in range(torch.cuda.device_count()):
                    allocated = torch.cuda.memory_allocated(i) / 1024**3
                    cached = torch.cuda.memory_reserved(i) / 1024**3
                    print(f"  GPU {i}: 已用 {allocated:.1f}GB, 缓存 {cached:.1f}GB")
        
        # 运行JOLT
        with torch.no_grad():  # 确保不计算梯度
            results = run_jolt(args=args, model=model, tokenizer=tokenizer)
        
        # 同步所有进程
        accelerator.wait_for_everyone()
        
        # 最终清理
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
        
        gc.collect()
        
        if accelerator.is_main_process:
            print("✓ 超级优化6GPU训练完成!")
        
        return True
        
    except torch.cuda.OutOfMemoryError as e:
        print(f"显存不足（即使在超级优化模式下）: {e}")
        print("建议:")
        print("  1. 进一步减小batch_size到1")
        print("  2. 减小train_size_limit到20以下")
        print("  3. 减小max_generated_length到10以下")
        print("  4. 考虑使用更小的模型")
        
        # 紧急清理显存
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                torch.cuda.set_device(i)
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()
        
        return False
        
    except Exception as e:
        print(f"超级优化运行错误: {e}")
        traceback.print_exc()
        return False

def main():
    """主函数"""
    try:
        print("启动超级显存优化6GPU JOLT训练")
        print("=" * 60)
        
        # 检查环境
        if 'LOCAL_RANK' in os.environ:
            local_rank = int(os.environ['LOCAL_RANK'])
            print(f"进程 {local_rank} 启动超级优化模式...")
        
        # 运行超级优化训练
        success = run_ultra_optimized_jolt()
        
        if not success:
            print("超级优化训练失败")
            sys.exit(1)
        else:
            print("超级优化训练成功完成！")
            
    except KeyboardInterrupt:
        print("训练被用户中断")
        sys.exit(1)
    except Exception as e:
        print(f"主函数错误: {e}")
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
