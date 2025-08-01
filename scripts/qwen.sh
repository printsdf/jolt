#!/bin/bash

# =============================================================================
# NOx预测自动化脚本 (最终离线版)
# 使用本地Llama 3 8B模型和优化的参数进行预测
# =============================================================================

echo "开始NOx预测任务"
echo "========================================"

# --- 配置参数 ---
EXPERIMENT_NAME="nox_qwen2_7b_shots_10_test"

LLM_TYPE="qwen2.5-7B-instruct"

# 2. 新增 LLM_PATH 变量，指向模型的实际文件夹路径
LLM_PATH="/data/models/Qwen/Qwen2.5-7B-Instruct-GPTQ-Int4"

# 2. 根据之前的实验，设置最佳实践参数 - 针对96GB显存优化
BATCH_SIZE=2             # 96GB显存可以支持稍大的批次
TRAIN_SIZE_LIMIT=80      # 大幅增加训练样本以显著降低MAE
TEST_SIZE_LIMIT=20       # 相应增加测试样本
NUM_SAMPLES=25           # 增加采样数提高预测稳定性

# 3. 使用更详细的“专家级”Prefix，因为Llama 3 8B能更好地理解复杂指令
PREFIX="This task is to predict the NOx reduction efficiency (target) of an industrial boiler's DeNOx system, likely an SNCR or SCR process. The target represents the percentage of NOx removed from flue gas. The prediction is based on time-series data from various sensors. The '氨水总流量' (ammonia water flow rate) is a critical input, as ammonia is the reactant for the NOx reduction. The '焚烧炉膛前温度' (furnace chamber temperature) is also vital, as the reduction reactions are highly effective only within a specific temperature window. Operational settings like '二次风机入口电动调节门位置反馈' (secondary air damper position) control the combustion stoichiometry and temperature, which in turn influences both initial NOx formation and the reduction process efficiency."

TARGET_COLUMN="target"
SEED=42
DATA_FILE="data/nox_prediction.csv"
OUTPUT_DIR="./output/nox_experiments"

# --- 预运行检查 ---
# 检查数据文件是否存在
if [ ! -f "$DATA_FILE" ]; then
    echo "错误: 数据文件 $DATA_FILE 不存在!"
    exit 1
fi
echo "数据文件检查通过: $DATA_FILE"

# 检查模型路径是否存在
if [ ! -d "$LLM_PATH" ]; then
    echo "错误: 模型路径 '$LLM_PATH' 不存在或不是一个目录!"
    echo "请在脚本中更新 LLM_TYPE 变量为您的正确模型路径。"
    exit 1
fi
echo "模型路径检查通过: $LLM_PATH"

# 创建输出目录
mkdir -p "$OUTPUT_DIR"
echo "输出目录已创建: $OUTPUT_DIR"

# --- 执行预测 ---
echo "开始执行NOx预测..."
python run_jolt.py \
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
  --train_end_index 1080 \
  --test_start_index 1080 \
  --test_end_index 1100 \
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
    exit 1
fi

# 放在训练/下载脚本最后一行
curl -s 'https://oapi.dingtalk.com/robot/send?access_token=979c5387ce734aea95b5b368629c8a412284dc2f52674dd0291eab5f154fb70a' \
  -H 'Content-Type: application/json' \
  -d "{\"msgtype\":\"text\",\"text\":{\"content\":\"任务 $$ 在 $(hostname) 完成于 $(date)\"}}"
