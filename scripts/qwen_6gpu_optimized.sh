#!/bin/bash

# =============================================================================
# NOx预测自动化脚本 (6GPU超级优化版本)
# 使用激进的显存优化策略，避免OOM
# =============================================================================

# 激进的显存优化设置
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,max_split_size_mb:128
export CUDA_LAUNCH_BLOCKING=0
export TORCH_USE_CUDA_DSA=1

# 分布式训练优化设置
export TORCH_DISTRIBUTED_DEBUG=INFO
export NCCL_DEBUG=WARN
export NCCL_TIMEOUT=3600
export NCCL_IB_DISABLE=1
export NCCL_P2P_DISABLE=1
export NCCL_TREE_THRESHOLD=0

echo "开始NOx预测任务 (6GPU超级优化模式)"
echo "========================================"

# --- 激进优化参数 ---
EXPERIMENT_NAME="nox_qwen2_7b_6gpu_ultra_optimized"

LLM_TYPE="qwen2.5-7B-instruct"
LLM_PATH="/data/models/Qwen/Qwen2.5-7B-Instruct-GPTQ-Int4"

# 超级保守的参数设置 - 最大化显存利用效率
BATCH_SIZE=1             # 每个GPU只处理1个样本
TRAIN_SIZE_LIMIT=30      # 大幅减少训练样本
TEST_SIZE_LIMIT=6        # 每个GPU只处理1个测试样本
NUM_SAMPLES=6            # 减少采样数
MAX_LENGTH=15            # 减少生成长度

# 简化的Prefix，减少token数量
PREFIX="Predict NOx emission based on industrial parameters."

TARGET_COLUMN="target"
SEED=42
DATA_FILE="data/nox_prediction.csv"
OUTPUT_DIR="./output/nox_experiments"

# --- 预运行优化 ---
echo "进行6GPU预运行优化..."

# 显示GPU状态
nvidia-smi --query-gpu=index,name,memory.total,memory.used,memory.free --format=csv

# 清理所有GPU内存
echo "清理所有GPU内存..."
python -c "
import torch
if torch.cuda.is_available():
    for i in range(torch.cuda.device_count()):
        torch.cuda.set_device(i)
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
    print(f'已清理 {torch.cuda.device_count()} 个GPU的内存')
"

# 检查文件
if [ ! -f "$DATA_FILE" ]; then
    echo "错误: 数据文件 $DATA_FILE 不存在!"
    exit 1
fi

if [ ! -d "$LLM_PATH" ]; then
    echo "错误: 模型路径 '$LLM_PATH' 不存在!"
    exit 1
fi

mkdir -p "$OUTPUT_DIR"

# --- 6GPU分布式执行 ---
echo "开始6GPU分布式训练..."
echo "每个GPU配置: batch_size=1, 显存优化=最大"

accelerate launch \
  --config_file accelerate_config_6gpu.yaml \
  --main_process_port 29500 \
  run_jolt_distributed.py \
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
  --train_end_index 1030 \
  --test_start_index 1030 \
  --test_end_index 1036 \
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

# 检查结果
if [ $? -eq 0 ]; then
    echo "========================================"
    echo "6GPU NOx预测任务完成!"
    echo "详细结果文件保存在: $OUTPUT_DIR"
    
    # 显示最终GPU状态
    echo "最终GPU状态:"
    nvidia-smi --query-gpu=index,memory.used,memory.free --format=csv
else
    echo "========================================"
    echo "6GPU预测任务执行失败!"
    echo "GPU状态诊断:"
    nvidia-smi
    
    echo "尝试清理GPU内存..."
    python -c "
import torch
if torch.cuda.is_available():
    for i in range(torch.cuda.device_count()):
        torch.cuda.set_device(i)
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
    print('GPU内存清理完成')
"
fi

# 最终清理
python -c "
import torch
import gc
gc.collect()
if torch.cuda.is_available():
    for i in range(torch.cuda.device_count()):
        torch.cuda.set_device(i)
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
print('最终清理完成')
"

# 通知完成
curl -s 'https://oapi.dingtalk.com/robot/send?access_token=979c5387ce734aea95b5b368629c8a412284dc2f52674dd0291eab5f154fb70a' \
  -H 'Content-Type: application/json' \
  -d "{\"msgtype\":\"text\",\"text\":{\"content\":\"6GPU任务 $EXPERIMENT_NAME 在 $(hostname) 完成于 $(date)\"}}"
