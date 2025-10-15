# Fine-tuning Performance Comparision
## Description
You need to fine-tune llama2-7B with different method on both v100 gpu and mi210 gpu for this assigment. Please run the model with lora, lora + ZeRO-2, lora + ZeRO-3, and qlora independently. Then, compare the results and answer the question.

## Envirionment
* creating a conda environment is recommended
* python ( >= 3.9)
* packages
    * torch >= 2.2
    * transformers >= 4.41 
    * datasets >= 2.19
    * accelerate >= 0.30
    * peft >= 0.11
    * bitsandbytes >= 0.43
    * deepspeed >= 0.14
    * trl >= 0.9
    * xformers >= 0.0.27
    * scipy
* cuda / rocm
> [!WARNING]  
> Some packages might not fit amd gpu, therefore you may like to try build from source.

## code 
[link](https://)

## parameters
These parameter should not be modified:
* batch size: 1
* sequence length: 350
* save steps: 100
* warmup_steps: 30
* LoRA rank: 64
* precision: v100 -> fp16, mi210 -> bf16
## example running command
```bash=
deepspeed --num_gpus 1 train.py \
  --model_name_or_path NousResearch/Llama-2-7b-chat-hf \
  --deepspeed ds_naive.json \
  --per_device_train_batch_size 1 \
  --gradient_accumulation_steps 1 \
  --learning_rate 1e-3 \
  --num_train_epochs 1 \
  --logging_steps 10 --save_steps 100 --warmup_steps 30 \
  --max_length 350 \
  --lora --lora_r 64 --lora_alpha 16 --lora_dropout 0.05 \
  --attn_impl eager --fp16
```

## Questions
1. 
2. 
## Report
In the report, you should include the result for 4 methods * 2 platform, the question above, and anything you would like to share regarding the assignment as well. Please only send the report in .pdf to eeclass.

