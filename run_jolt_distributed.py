#!/usr/bin/env python3
"""
支持分布式训练的JOLT运行脚本
解决ChildFailedError问题
"""

import os
import sys
import torch
import torch.distributed as dist
from accelerate import Accelerator
from accelerate.utils import set_seed
import traceback

def setup_distributed():
    """设置分布式训练环境"""
    try:
        accelerator = Accelerator()
        
        # 只在主进程打印信息
        if accelerator.is_main_process:
            print(f"分布式训练设置:")
            print(f"  设备: {accelerator.device}")
            print(f"  进程数: {accelerator.num_processes}")
            print(f"  本地进程索引: {accelerator.local_process_index}")
            print(f"  是否主进程: {accelerator.is_main_process}")
            print(f"  混合精度: {accelerator.mixed_precision}")
        
        return accelerator
        
    except Exception as e:
        print(f"分布式设置失败: {e}")
        traceback.print_exc()
        return None

def run_jolt_distributed():
    """运行分布式JOLT训练"""
    try:
        # 设置分布式环境
        accelerator = setup_distributed()
        if accelerator is None:
            return False
        
        # 导入JOLT模块
        from run_jolt import run_jolt
        from parse_args import parse_command_line
        from hf_api import get_model_and_tokenizer
        
        # 解析命令行参数
        args = parse_command_line()
        
        # 设置随机种子
        set_seed(args.seed)
        
        # 只在主进程打印参数信息
        if accelerator.is_main_process:
            print(f"实验参数:")
            print(f"  实验名称: {args.experiment_name}")
            print(f"  模型类型: {args.llm_type}")
            print(f"  模型路径: {args.llm_path}")
            print(f"  批次大小: {args.batch_size}")
            print(f"  训练样本数: {args.train_size_limit}")
            print(f"  测试样本数: {args.test_size_limit}")
        
        # 加载模型和分词器
        if accelerator.is_main_process:
            print("正在加载模型...")
        
        model, tokenizer = get_model_and_tokenizer(args)
        
        # 使用accelerator准备模型
        model = accelerator.prepare(model)
        
        if accelerator.is_main_process:
            print("模型加载完成，开始训练...")
        
        # 运行JOLT
        results = run_jolt(args=args, model=model, tokenizer=tokenizer)
        
        # 等待所有进程完成
        accelerator.wait_for_everyone()
        
        if accelerator.is_main_process:
            print("✓ 分布式训练完成!")
        
        return True
        
    except torch.cuda.OutOfMemoryError as e:
        print(f"GPU内存不足: {e}")
        print("建议:")
        print("  1. 减小batch_size")
        print("  2. 减小train_size_limit和test_size_limit")
        print("  3. 使用梯度累积")
        return False
        
    except Exception as e:
        print(f"运行时错误: {e}")
        traceback.print_exc()
        return False

def main():
    """主函数"""
    try:
        # 检查分布式环境变量
        if 'LOCAL_RANK' in os.environ:
            local_rank = int(os.environ['LOCAL_RANK'])
            print(f"进程 {local_rank} 开始运行...")
        
        # 运行分布式训练
        success = run_jolt_distributed()
        
        if not success:
            sys.exit(1)
            
    except KeyboardInterrupt:
        print("训练被用户中断")
        sys.exit(1)
    except Exception as e:
        print(f"主函数错误: {e}")
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
