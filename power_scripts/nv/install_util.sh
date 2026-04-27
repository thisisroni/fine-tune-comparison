# Use your conda / venv / uv env
salloc --account=ACD110018 \
       --partition=gp1d \
       --job-name=run \
       --nodes=1 \
       --ntasks-per-node=1 \
       --cpus-per-task=4 \
       --gpus-per-node=1 \
       --time=02:00:00

# conda create -n finetune python=3.10 -y
conda activate finetune
pip install -r llama/fine-tune-comparison/requirements.txt
## 要改 requirements.txt
pip install torch==2.5.1+cu121 \
  --index-url https://download.pytorch.org/whl/cu121
pip install nvidia-ml-py pandas matplotlib
