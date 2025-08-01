#!/bin/bash

# =============================================================================
# NOx预测自动化脚本 (稳定多GPU版本)
# 使用保守的分布式训练配置，避免ChildFailedError
# =============================================================================

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# 分布式训练调试设置
export TORCH_DISTRIBUTED_DEBUG=INFO
export NCCL_DEBUG=INFO
export NCCL_TIMEOUT=1800
export NCCL_IB_DISABLE=1
export NCCL_P2P_DISABLE=1

echo "开始NOx预测任务 (稳定多GPU模式)"
echo "========================================"

# --- 配置参数 ---
EXPERIMENT_NAME="nox_qwen2_7b_stable_multigpu"

LLM_TYPE="qwen2.5-7B-instruct"

# 模型路径
LLM_PATH="/data/models/Qwen/Qwen2.5-7B-Instruct-GPTQ-Int4"

# 保守的参数设置，确保稳定性
BATCH_SIZE=1             # 每个GPU的批次大小
TRAIN_SIZE_LIMIT=40      # 减少训练样本数，降低内存压力
TEST_SIZE_LIMIT=10       # 减少测试样本数
NUM_SAMPLES=10           # 减少采样数

# 专家级Prefix
PREFIX="You are an expert in environmental engineering and NOx emission prediction. Based on the given operational parameters of industrial equipment, predict the NOx emission levels with high accuracy. Consider the complex relationships between temperature, pressure, flow rates, and chemical compositions."

TARGET_COLUMN="target"
SEED=42
DATA_FILE="data/nox_prediction.csv"
OUTPUT_DIR="./output/nox_experiments"

# --- 预运行检查 ---
echo "进行预运行检查..."

# 检查GPU状态
nvidia-smi
echo "GPU检查完成"

# 检查数据文件是否存在
if [ ! -f "$DATA_FILE" ]; then
    echo "错误: 数据文件 $DATA_FILE 不存在!"
    exit 1
fi
echo "数据文件检查通过: $DATA_FILE"

# 检查模型路径是否存在
if [ ! -d "$LLM_PATH" ]; then
    echo "错误: 模型路径 '$LLM_PATH' 不存在或不是一个目录!"
    echo "请在脚本中更新 LLM_PATH 变量为您的正确模型路径。"
    exit 1
fi
echo "模型路径检查通过: $LLM_PATH"

# 创建输出目录
mkdir -p "$OUTPUT_DIR"
echo "输出目录已创建: $OUTPUT_DIR"

# 清理GPU内存
echo "清理GPU内存..."
python -c "import torch; torch.cuda.empty_cache(); print('GPU内存已清理')"

# --- 执行预测 ---
echo "开始执行NOx预测 (保守多GPU模式)..."
echo "使用2个GPU进行分布式训练..."

accelerate launch \
  --config_file accelerate_config_conservative.yaml \
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
  --train_end_index 1040 \
  --test_start_index 1040 \
  --test_end_index 1050 \
  --mode sample_logpy \
  --y_column_names "$TARGET_COLUMN" \
  --y_column_types numerical \
  --num_decimal_places_x 2 \
  --num_decimal_places_y 1 \
  --max_generated_length 20 \
  --header_option headers_as_item_prefix \
  --test_fraction 0.2 \
  --seed $SEED \
  --num_samples $NUM_SAMPLES \
  --temperature 0.3 \
  --prefix "$PREFIX"

# 检查执行结果
if [ $? -eq 0 ]; then
    echo "========================================"
    echo "NOx预测任务完成!"
    echo "详细结果文件保存在: $OUTPUT_DIR"
else
    echo "========================================"
    echo "预测任务执行失败!"
    echo "请检查以上错误信息并重试"
    
    # 显示GPU状态
    echo "当前GPU状态:"
    nvidia-smi
fi

# 清理GPU内存
python -c "import torch; torch.cuda.empty_cache(); print('训练后GPU内存已清理')"

# 通知完成
curl -s 'https://oapi.dingtalk.com/robot/send?access_token=979c5387ce734aea95b5b368629c8a412284dc2f52674dd0291eab5f154fb70a' \
  -H 'Content-Type: application/json' \
  -d "{\"msgtype\":\"text\",\"text\":{\"content\":\"任务 $EXPERIMENT_NAME 在 $(hostname) 完成于 $(date)\"}}"
