#!/usr/bin/env python3
"""
带错误处理的JOLT运行脚本
用于捕获和诊断运行时错误
"""

import sys
import os
import traceback
import torch
from pathlib import Path

def setup_environment():
    """设置环境变量"""
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'  # 强制单GPU
    
    print("环境设置:")
    print(f"  PYTORCH_CUDA_ALLOC_CONF: {os.environ.get('PYTORCH_CUDA_ALLOC_CONF')}")
    print(f"  CUDA_VISIBLE_DEVICES: {os.environ.get('CUDA_VISIBLE_DEVICES')}")

def check_gpu_memory():
    """检查GPU内存状态"""
    if torch.cuda.is_available():
        device = torch.cuda.current_device()
        total_memory = torch.cuda.get_device_properties(device).total_memory / 1024**3
        allocated = torch.cuda.memory_allocated(device) / 1024**3
        cached = torch.cuda.memory_reserved(device) / 1024**3
        
        print(f"\nGPU内存状态:")
        print(f"  设备: {torch.cuda.get_device_name(device)}")
        print(f"  总内存: {total_memory:.1f} GB")
        print(f"  已分配: {allocated:.1f} GB")
        print(f"  缓存: {cached:.1f} GB")
        print(f"  可用: {total_memory - cached:.1f} GB")
        
        return total_memory - cached > 5.0  # 至少需要5GB可用内存
    return False

def run_jolt_with_monitoring():
    """运行JOLT并监控内存使用"""
    try:
        # 导入必要的模块
        from run_jolt import main as jolt_main
        from parse_args import parse_command_line
        
        print("开始运行JOLT...")
        
        # 设置命令行参数
        sys.argv = [
            'run_jolt.py',
            '--experiment_name', 'nox_qwen2_7b_shots_10_single_gpu',
            '--data_path', 'data/nox_prediction.csv',
            '--llm_type', 'qwen2.5-7B-instruct',
            '--llm_path', '/data/models/Qwen/Qwen2.5-7B-Instruct-GPTQ-Int4',
            '--output_dir', './output/nox_experiments',
            '--batch_size', '1',
            '--train_size_limit', '80',
            '--test_size_limit', '20',
            '--csv_split_option', 'fixed_indices',
            '--train_start_index', '1000',
            '--train_end_index', '1080',
            '--test_start_index', '1080',
            '--test_end_index', '1100',
            '--mode', 'sample_logpy',
            '--y_column_names', 'target',
            '--y_column_types', 'numerical',
            '--num_decimal_places_x', '2',
            '--num_decimal_places_y', '1',
            '--max_generated_length', '20',
            '--header_option', 'headers_as_item_prefix',
            '--test_fraction', '0.2',
            '--seed', '42',
            '--num_samples', '25',
            '--temperature', '0.3',
            '--prefix', 'You are an expert in environmental engineering and NOx emission prediction.'
        ]
        
        # 运行主函数
        jolt_main()
        
        print("✓ JOLT运行成功完成!")
        return True
        
    except torch.cuda.OutOfMemoryError as e:
        print(f"✗ GPU内存不足错误: {e}")
        print("建议:")
        print("  1. 减小batch_size")
        print("  2. 减小train_size_limit和test_size_limit")
        print("  3. 减小num_samples")
        return False
        
    except ImportError as e:
        print(f"✗ 模块导入错误: {e}")
        print("请检查Python环境和依赖包")
        return False
        
    except FileNotFoundError as e:
        print(f"✗ 文件未找到错误: {e}")
        print("请检查模型路径和数据文件路径")
        return False
        
    except Exception as e:
        print(f"✗ 运行时错误: {e}")
        print("详细错误信息:")
        traceback.print_exc()
        return False

def main():
    """主函数"""
    print("JOLT 错误处理运行器")
    print("=" * 50)
    
    # 设置环境
    setup_environment()
    
    # 检查GPU内存
    if not check_gpu_memory():
        print("⚠️ GPU内存可能不足，建议减小批次大小")
    
    # 运行JOLT
    success = run_jolt_with_monitoring()
    
    if success:
        print("\n🎉 任务成功完成!")
    else:
        print("\n❌ 任务失败，请查看上述错误信息")
        sys.exit(1)

if __name__ == "__main__":
    main()
