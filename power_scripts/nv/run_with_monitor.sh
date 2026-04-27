#!/bin/bash

# 1. 確保 log 目錄存在
# Use your conda / venv / uv env
module use /opt/nvidia/hpc_sdk/modulefiles
ml nvhpc-byo-compiler/21.7
source /opt/miniconda3/etc/profile.d/conda.sh
conda activate finetune

# Change this to your working / logs directory
WORK_DIR=$HOME/llm-inference/lch/fine-tune-comparison
SUB_LOG_DIR=test
LOG_NAME=finetune
LOG_DIR=$WORK_DIR/logs/${SUB_LOG_DIR}_$(date +"%Y-%m-%d_%H:%M:%S")


cd $WORK_DIR/power_scripts/nv

mkdir -p $LOG_DIR

# 2. 啟動 Python 監控 (背景執行)
python3 monitor_nv.py $LOG_DIR & MONITOR_PID=$!

echo "[$(date)] 啟動監控 (PID: $MONITOR_PID) 並執行程式..."

cd $WORK_DIR

# 3. 執行指令
deepspeed --num_gpus 1 $WORK_DIR/train.py \
  --model_name_or_path /home/lsalab/llm-inference/lch/Meta-Llama-3-8B-Instruct \
  --deepspeed ds_z2.json \
  --per_device_train_batch_size 1 \
  --gradient_accumulation_steps 1 \
  --learning_rate 1e-3 \
  --num_train_epochs 1 \
  --logging_steps 10 \
  --save_steps 100 \
  --warmup_steps 30 \
  --max_length 350 \
  --lora \
  --lora_r 64 \
  --lora_alpha 16 \
  --lora_dropout 0.05 \
  --attn_impl eager \
  --fp16 \
  2>&1 | tee $LOG_DIR/lora_zero2_$(date +"%Y-%m-%d_%H:%M:%S").log
    

# 4. 程式結束後，通知監控腳本停止執行
echo "[$(date)] 程式執行完畢，正在停止監控並產生圖表..."
kill -TERM $MONITOR_PID

# 等待監控腳本完成寫入與繪圖
wait $MONITOR_PID

echo "End"
