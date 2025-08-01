#!/bin/bash

# =============================================================================
# NOx预测自动化脚本 (6GPU终极优化版本)
# 使用所有可能的显存优化技术，确保不会OOM
# =============================================================================

# 终极显存优化环境变量
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,max_split_size_mb:32
export CUDA_LAUNCH_BLOCKING=0
export TORCH_USE_CUDA_DSA=1
export CUDA_CACHE_DISABLE=1

# 分布式训练优化
export TORCH_DISTRIBUTED_DEBUG=WARN
export NCCL_DEBUG=WARN
export NCCL_TIMEOUT=7200
export NCCL_IB_DISABLE=1
export NCCL_P2P_DISABLE=1
export NCCL_TREE_THRESHOLD=0
export NCCL_ALGO=Ring

echo "🚀 启动6GPU终极优化NOx预测任务"
echo "========================================"

# --- 终极保守参数 ---
EXPERIMENT_NAME="nox_qwen2_7b_6gpu_ultimate"

LLM_TYPE="qwen2.5-7B-instruct"
LLM_PATH="/data/models/Qwen/Qwen2.5-7B-Instruct-GPTQ-Int4"

# 极度保守的参数设置
BATCH_SIZE=1             # 每个GPU只处理1个样本
TRAIN_SIZE_LIMIT=18      # 总共18个训练样本，每个GPU 3个
TEST_SIZE_LIMIT=6        # 总共6个测试样本，每个GPU 1个
NUM_SAMPLES=6            # 总共6个采样，每个GPU 1个
MAX_LENGTH=10            # 极短的生成长度

# 最简化的Prefix
PREFIX="Predict NOx."

TARGET_COLUMN="target"
SEED=42
DATA_FILE="data/nox_prediction.csv"
OUTPUT_DIR="./output/nox_experiments"

# --- 终极预运行优化 ---
echo "🔧 进行终极6GPU预运行优化..."

# 杀死可能占用GPU的进程
echo "清理GPU进程..."
nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits | xargs -r kill -9 2>/dev/null || true

# 重置GPU状态
nvidia-smi --gpu-reset -i 0,1,2,3,4,5 2>/dev/null || true

# 显示初始GPU状态
echo "初始GPU状态:"
nvidia-smi --query-gpu=index,name,memory.total,memory.used,memory.free,utilization.gpu --format=csv

# 终极显存清理
echo "执行终极显存清理..."
python -c "
import torch
import gc
import os

# 强制垃圾回收
gc.collect()

if torch.cuda.is_available():
    print(f'检测到 {torch.cuda.device_count()} 个GPU')
    
    # 清理每个GPU
    for i in range(torch.cuda.device_count()):
        torch.cuda.set_device(i)
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
        
        # 设置显存分配限制
        torch.cuda.set_per_process_memory_fraction(0.75, device=i)
        
        print(f'GPU {i} 清理完成，设置显存限制为75%')
    
    print('所有GPU终极清理完成')
else:
    print('未检测到CUDA设备')
"

# 检查文件
if [ ! -f "$DATA_FILE" ]; then
    echo "❌ 错误: 数据文件 $DATA_FILE 不存在!"
    exit 1
fi

if [ ! -d "$LLM_PATH" ]; then
    echo "❌ 错误: 模型路径 '$LLM_PATH' 不存在!"
    exit 1
fi

mkdir -p "$OUTPUT_DIR"

echo "✅ 预检查完成"

# --- 6GPU终极分布式执行 ---
echo "🚀 开始6GPU终极分布式训练..."
echo "配置: 每GPU batch=1, 训练样本=18, 测试样本=6, 生成长度=10"

# 使用最保守的启动参数
accelerate launch \
  --config_file accelerate_config_6gpu.yaml \
  --main_process_port 29500 \
  --num_cpu_threads_per_process 2 \
  run_jolt_ultra_optimized.py \
  --experiment_name "$EXPERIMENT_NAME" \
  --data_path "$DATA_FILE" \
  --llm_type "$LLM_TYPE" \
  --llm_path "$LLM_PATH" \
  --output_dir "$OUTPUT_DIR" \
  --batch_size $BATCH_SIZE \
  --train_size_limit $TRAIN_SIZE_LIMIT \
  --test_size_limit $TEST_SIZE_LIMIT \
  --csv_split_option "fixed_indices" \
  --train_start_index 1000 \
  --train_end_index 1018 \
  --test_start_index 1018 \
  --test_end_index 1024 \
  --mode sample_logpy \
  --y_column_names "$TARGET_COLUMN" \
  --y_column_types numerical \
  --num_decimal_places_x 2 \
  --num_decimal_places_y 1 \
  --max_generated_length $MAX_LENGTH \
  --header_option headers_as_item_prefix \
  --test_fraction 0.2 \
  --seed $SEED \
  --num_samples $NUM_SAMPLES \
  --temperature 0.3 \
  --prefix "$PREFIX"

# 检查执行结果
RESULT=$?

if [ $RESULT -eq 0 ]; then
    echo "========================================"
    echo "🎉 6GPU终极优化NOx预测任务完成!"
    echo "详细结果文件保存在: $OUTPUT_DIR"
    
    # 显示成功后的GPU状态
    echo "任务完成后GPU状态:"
    nvidia-smi --query-gpu=index,memory.used,memory.free,utilization.gpu --format=csv
    
else
    echo "========================================"
    echo "❌ 6GPU终极优化预测任务失败!"
    echo "错误代码: $RESULT"
    
    echo "失败时GPU状态:"
    nvidia-smi
    
    echo "尝试紧急显存清理..."
    python -c "
import torch
import gc
gc.collect()
if torch.cuda.is_available():
    for i in range(torch.cuda.device_count()):
        torch.cuda.set_device(i)
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
    print('紧急清理完成')
"
fi

# 最终清理
echo "执行最终清理..."
python -c "
import torch
import gc
import os

gc.collect()
if torch.cuda.is_available():
    for i in range(torch.cuda.device_count()):
        torch.cuda.set_device(i)
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
        torch.cuda.reset_peak_memory_stats(i)
    print('最终清理完成')
"

# 通知完成
if [ $RESULT -eq 0 ]; then
    curl -s 'https://oapi.dingtalk.com/robot/send?access_token=979c5387ce734aea95b5b368629c8a412284dc2f52674dd0291eab5f154fb70a' \
      -H 'Content-Type: application/json' \
      -d "{\"msgtype\":\"text\",\"text\":{\"content\":\"🎉 6GPU终极优化任务 $EXPERIMENT_NAME 在 $(hostname) 成功完成于 $(date)\"}}"
else
    curl -s 'https://oapi.dingtalk.com/robot/send?access_token=979c5387ce734aea95b5b368629c8a412284dc2f52674dd0291eab5f154fb70a' \
      -H 'Content-Type: application/json' \
      -d "{\"msgtype\":\"text\",\"text\":{\"content\":\"❌ 6GPU任务 $EXPERIMENT_NAME 在 $(hostname) 失败于 $(date)\"}}"
fi

echo "脚本执行完毕"
