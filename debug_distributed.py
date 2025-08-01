#!/usr/bin/env python3
"""
分布式训练调试脚本
用于诊断JOLT项目中的ChildFailedError问题
"""

import torch
import os
import sys
from pathlib import Path

def check_cuda_setup():
    """检查CUDA设置"""
    print("=== CUDA 设置检查 ===")
    print(f"PyTorch版本: {torch.__version__}")
    print(f"CUDA可用: {torch.cuda.is_available()}")
    
    if torch.cuda.is_available():
        print(f"CUDA版本: {torch.version.cuda}")
        print(f"GPU数量: {torch.cuda.device_count()}")
        
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            print(f"GPU {i}: {props.name}")
            print(f"  显存: {props.total_memory / 1024**3:.1f} GB")
            print(f"  计算能力: {props.major}.{props.minor}")
            
            # 检查显存使用情况
            torch.cuda.set_device(i)
            allocated = torch.cuda.memory_allocated(i) / 1024**3
            cached = torch.cuda.memory_reserved(i) / 1024**3
            print(f"  已分配显存: {allocated:.1f} GB")
            print(f"  缓存显存: {cached:.1f} GB")
    else:
        print("CUDA不可用!")
        return False
    
    return True

def check_model_path():
    """检查模型路径"""
    print("\n=== 模型路径检查 ===")
    model_path = "/data/models/Qwen/Qwen2.5-7B-Instruct-GPTQ-Int4"
    
    if os.path.exists(model_path):
        print(f"✓ 模型路径存在: {model_path}")
        
        # 检查关键文件
        key_files = ["config.json", "tokenizer.json", "tokenizer_config.json"]
        for file in key_files:
            file_path = os.path.join(model_path, file)
            if os.path.exists(file_path):
                print(f"✓ {file} 存在")
            else:
                print(f"✗ {file} 缺失")
        
        # 检查模型文件
        model_files = [f for f in os.listdir(model_path) if f.endswith(('.bin', '.safetensors', '.pt'))]
        if model_files:
            print(f"✓ 找到模型文件: {model_files}")
        else:
            print("✗ 未找到模型文件")
            
    else:
        print(f"✗ 模型路径不存在: {model_path}")
        return False
    
    return True

def check_data_file():
    """检查数据文件"""
    print("\n=== 数据文件检查 ===")
    data_file = "data/nox_prediction.csv"
    
    if os.path.exists(data_file):
        print(f"✓ 数据文件存在: {data_file}")
        
        # 检查文件大小
        size = os.path.getsize(data_file) / 1024
        print(f"  文件大小: {size:.1f} KB")
        
        # 检查前几行
        try:
            with open(data_file, 'r') as f:
                lines = f.readlines()[:5]
                print(f"  总行数: {len(lines)} (显示前5行)")
                for i, line in enumerate(lines):
                    print(f"    {i+1}: {line.strip()[:100]}...")
        except Exception as e:
            print(f"  读取文件时出错: {e}")
            
    else:
        print(f"✗ 数据文件不存在: {data_file}")
        return False
    
    return True

def test_model_loading():
    """测试模型加载"""
    print("\n=== 模型加载测试 ===")
    
    try:
        from hf_api import get_model_and_tokenizer
        from parse_args import init_option_parser
        
        # 创建测试参数
        parser = init_option_parser()
        test_args = parser.parse_args([
            "--llm_type", "qwen2.5-7B-instruct",
            "--llm_path", "/data/models/Qwen/Qwen2.5-7B-Instruct-GPTQ-Int4",
            "--batch_size", "1"
        ])
        
        print("正在加载模型...")
        model, tokenizer = get_model_and_tokenizer(test_args)
        print("✓ 模型加载成功!")
        
        # 测试简单推理
        test_input = "Hello, how are you?"
        inputs = tokenizer(test_input, return_tensors="pt")
        
        if torch.cuda.is_available():
            inputs = {k: v.to(model.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = model.generate(**inputs, max_new_tokens=10)
        
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        print(f"✓ 推理测试成功: {response}")
        
        return True
        
    except Exception as e:
        print(f"✗ 模型加载失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def check_accelerate_config():
    """检查accelerate配置"""
    print("\n=== Accelerate 配置检查 ===")
    
    try:
        from accelerate import Accelerator
        from accelerate.utils import gather_object
        
        accelerator = Accelerator()
        print(f"✓ Accelerator初始化成功")
        print(f"  设备: {accelerator.device}")
        print(f"  进程数: {accelerator.num_processes}")
        print(f"  本地进程数: {accelerator.local_process_index}")
        print(f"  是否主进程: {accelerator.is_main_process}")
        
        return True
        
    except Exception as e:
        print(f"✗ Accelerate配置有问题: {e}")
        return False

def main():
    """主函数"""
    print("JOLT 分布式训练诊断工具")
    print("=" * 50)
    
    checks = [
        ("CUDA设置", check_cuda_setup),
        ("模型路径", check_model_path),
        ("数据文件", check_data_file),
        ("Accelerate配置", check_accelerate_config),
        ("模型加载", test_model_loading),
    ]
    
    results = {}
    for name, check_func in checks:
        try:
            results[name] = check_func()
        except Exception as e:
            print(f"✗ {name}检查时出错: {e}")
            results[name] = False
    
    print("\n" + "=" * 50)
    print("诊断结果汇总:")
    
    all_passed = True
    for name, passed in results.items():
        status = "✓ 通过" if passed else "✗ 失败"
        print(f"  {name}: {status}")
        if not passed:
            all_passed = False
    
    if all_passed:
        print("\n🎉 所有检查都通过了！可以尝试运行训练。")
    else:
        print("\n⚠️  发现问题，请根据上述信息进行修复。")
    
    print("\n建议的运行方式:")
    print("1. 如果所有检查通过，使用: bash scripts/qwen.sh")
    print("2. 如果有分布式问题，使用: bash scripts/qwen_single_gpu.sh")
    print("3. 如果模型加载有问题，检查模型路径和权限")

if __name__ == "__main__":
    main()
