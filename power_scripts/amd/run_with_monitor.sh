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
ml miniconda3/conda24.5.0_py3.9 cuda/12.8
conda activate vllm-xformers-env

# Change this to your working / logs directory
WORK_DIR=/work/jonathan0hsu/llm-inference/paper
SUB_LOG_DIR=vllm_bench
LOG_NAME=bench_vllm_8b
LOG_DIR=$WORK_DIR/logs/${SUB_LOG_DIR}_$(date +"%Y-%m-%d_%H:%M:%S")


cd $WORK_DIR

mkdir -p $LOG_DIR

# 2. 啟動 Python 監控 (背景執行)
python3 monitor_amd.py $LOG_DIR & MONITOR_PID=$!

echo "[$(date)] 啟動監控 (PID: $MONITOR_PID) 並執行程式..."

# 3. 執行指令

./run_ap.sh 2>&1 | tee $LOG_DIR/${LOG_NAME}_$(date +"%Y-%m-%d_%H:%M:%S").log\
    

# 4. 程式結束後，通知監控腳本停止執行
echo "[$(date)] 程式執行完畢，正在停止監控並產生圖表..."
kill -TERM $MONITOR_PID

# 等待監控腳本完成寫入與繪圖
wait $MONITOR_PID

echo "End"
