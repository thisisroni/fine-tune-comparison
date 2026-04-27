#!/bin/bash
#SBATCH -p gtest 
#SBATCH --account=ACD110018 
#SBATCH --nodes=1 
#SBATCH --ntasks-per-node=1 
#SBATCH --time=00:30:00 
#SBATCH --cpus-per-task=4 
#SBATCH --gpus-per-node=1 
#SBATCH --mem=90G 


# 1. 確保 log 目錄存在
# Use your conda / venv / uv env
ml biology/miniconda/miniconda3 biology/cuda/12.0.0 compiler/gcc/10.4.0
conda activate lora

# Change this to your working / logs directory
WORK_DIR=$HOME/llama/fine-tune-comparison
SUB_LOG_DIR=test
LOG_NAME=finetune
LOG_DIR=$WORK_DIR/logs/${SUB_LOG_DIR}_$(date +"%Y-%m-%d_%H:%M:%S")


cd $WORK_DIR

mkdir -p $LOG_DIR

# 2. 啟動 Python 監控 (背景執行)
python3 monitor_nv.py $LOG_DIR & MONITOR_PID=$!

echo "[$(date)] 啟動監控 (PID: $MONITOR_PID) 並執行程式..."

# 3. 執行指令
deepspeed --num_gpus 1 train.py \
  --model_name_or_path /work/jonathan0hsu/llm-inference/model/Llama-3-8B-Instruct \
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
