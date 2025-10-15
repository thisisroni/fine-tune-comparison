#!/usr/bin/env python3
import argparse, os
from packaging import version

import torch
from datasets import load_dataset

from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
from peft import LoraConfig, get_peft_model
import trl

from itertools import islice
import numpy as np

# --- 解析參數 ---
# --- 解析參數 ---
def get_args():
    p = argparse.ArgumentParser()
    # 你原本的
    p.add_argument("--base_model", type=str, default="NousResearch/Llama-2-7b-chat-hf")
    p.add_argument("--dataset", type=str, default="mlabonne/guanaco-llama2-1k")
    p.add_argument("--output_dir", type=str, default="./results_modified")
    p.add_argument("--epochs", type=float, default=1.0)
    p.add_argument("--per_device_bs", type=int, default=1)
    p.add_argument("--grad_accum", type=int, default=1)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight_decay", type=float, default=0.0)
    p.add_argument("--warmup_steps", type=int, default=0, help="Number of warmup steps for learning rate scheduler")
    p.add_argument("--warmup_ratio", type=float, default=0.0, help="Warmup ratio of total training steps")
    p.add_argument("--max_length", type=int, default=350)
    p.add_argument("--logging_steps", type=int, default=20)
    p.add_argument("--save_steps", type=int, default=200)
    p.add_argument("--packing", action="store_true")
    p.add_argument("--use_flash_attn2", action="store_true")
    p.add_argument("--qlora", action="store_true")
    p.add_argument("--bf16", action="store_true")
    p.add_argument("--fp16", action="store_true")
    p.add_argument("--merge_and_save", action="store_true")
    p.add_argument("--seed", type=int, default=42)

    # === 新增：DeepSpeed 與別名對齊 ===
    p.add_argument("--deepspeed", type=str, default=None, help="Path to ds config json")
    p.add_argument("--local_rank", type=int, default=-1, help="For DeepSpeed/torchrun")

    # 別名（讓 deepspeed 指令可用）
    p.add_argument("--model_name_or_path", type=str, help="alias of --base_model")
    p.add_argument("--per_device_train_batch_size", type=int, help="alias of --per_device_bs")
    p.add_argument("--gradient_accumulation_steps", type=int, help="alias of --grad_accum")
    p.add_argument("--num_train_epochs", type=float, help="alias of --epochs")
    p.add_argument("--learning_rate", type=float, help="alias of --lr")
    p.add_argument("--max_seq_length", type=int, help="alias of --max_length")

    # LoRA 參數（原本沒有顯式，補上）
    p.add_argument("--lora", action="store_true")
    p.add_argument("--lora_r", type=int, default=64)
    p.add_argument("--lora_alpha", type=int, default=16)
    p.add_argument("--lora_dropout", type=float, default=0.05)

    # FlashAttention 控制（字串型態更彈性）
    p.add_argument("--attn_impl", type=str, default="eager",
                   choices=["eager", "flash_attn2"])

    args = p.parse_args()

    # 把別名灌回正名
    if args.model_name_or_path: args.base_model = args.model_name_or_path
    if args.per_device_train_batch_size: args.per_device_bs = args.per_device_train_batch_size
    if args.gradient_accumulation_steps: args.grad_accum = args.gradient_accumulation_steps
    if args.num_train_epochs: args.epochs = args.num_train_epochs
    if args.learning_rate: args.lr = args.learning_rate
    if args.max_seq_length: args.max_length = args.max_seq_length

    return args


def main():
    args = get_args()
    os.makedirs(args.output_dir, exist_ok=True)

    torch.manual_seed(args.seed)

    # 設定 tokenizer 平行化避免警告
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    # --- Tokenizer ---
    tokenizer = AutoTokenizer.from_pretrained(args.base_model, use_fast=True, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    # --- Model loading options ---
    dtype = None
    if args.bf16 and torch.cuda.is_available() and torch.cuda.is_bf16_supported():
        dtype = torch.bfloat16
    elif args.fp16:
        dtype = torch.float16

    bnb_config = None
    if args.qlora:
        from transformers import BitsAndBytesConfig
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16 if args.bf16 else torch.float16,
        )

    # === DeepSpeed 模型載入關鍵修正 ===
    from_kwargs = dict(
        low_cpu_mem_usage=True,
        torch_dtype=(dtype if dtype is not None else "auto"),
        quantization_config=bnb_config,     # QLoRA 時生效；LoRA baseline 則為 None
    )
    
    if args.deepspeed is None:
        # 沒用 DeepSpeed 時才用 device_map="auto"
        from_kwargs["device_map"] = "auto"
    else:
        # DeepSpeed 時：單節點多卡時，每個 rank 綁自己那張卡（避免跨卡上下文）
        from_kwargs["device_map"] = {"": 0}

    model = AutoModelForCausalLM.from_pretrained(args.base_model, **from_kwargs)
    model.config.use_cache = False

    # 訓練端 FlashAttention（可用時）
    if args.attn_impl == "flash_attn2":
        try:
            model.config.attn_implementation = "flash_attention_2"
        except Exception:
            pass

    # --- Dataset ---
    train_ds = load_dataset(args.dataset, split="train")

    def estimate_avg_len(ds, tok, max_len=350, sample=256):
        lens = []
        for ex in islice(ds, 0, sample):
            ids = tok(ex["text"], max_length=max_len, truncation=True, add_special_tokens=True)["input_ids"]
            lens.append(len(ids))
        return float(np.mean(lens))

    avg_len = estimate_avg_len(train_ds, tokenizer, args.max_length, 256)
    print(f"[INFO] estimated avg seq len = {avg_len:.1f}")

    # --- LoRA 設定 ---
    peft_cfg = None
    if args.lora or args.qlora:
        peft_cfg = LoraConfig(
            r=args.lora_r,                    # ←吃 CLI 參數（預設 64）
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=[
                "q_proj","k_proj","v_proj","o_proj",
                "gate_proj","up_proj","down_proj"
            ],
        )
        model = get_peft_model(model, peft_cfg)

    # 快速核對 LoRA 有啟用的小撇步
    if hasattr(model, "print_trainable_parameters"):
        model.print_trainable_parameters()

    # === 1) 先把資料 tokenize 成固定 350 tokens（和論文一致） ===
    from datasets import DatasetDict
    from transformers import DataCollatorForLanguageModeling

    def to_text(example):
        # guanaco-llama2-1k 有 "text" 欄位，保險起見做 fallback
        if "text" in example and isinstance(example["text"], str):
            return example["text"]
        # 把所有欄位串成一行
        return " ".join(str(v) for v in example.values())

    def tokenize_fn(example):
        txt = to_text(example)
        toks = tokenizer(
            txt,
            truncation=True,
            padding="max_length",
            max_length=args.max_length,
            return_tensors=None,
        )
        return toks

    # map 成 tokenized dataset（移除原欄位，避免多餘拷貝）
    tokenized = train_ds.map(tokenize_fn, remove_columns=train_ds.column_names, batched=False)

    # causal LM 的 collator（不是 MLM）
    collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    # === 2) TrainingArguments 已經建好，直接用 HF Trainer ===
    train_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.per_device_bs,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        num_train_epochs=args.epochs,
        warmup_steps=args.warmup_steps,
        warmup_ratio=args.warmup_ratio,
        weight_decay=args.weight_decay,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        bf16=(dtype == torch.bfloat16),
        fp16=(dtype == torch.float16),
        deepspeed=args.deepspeed,     # ←關鍵
        report_to=[],
        save_strategy="no",            # ← 不要在訓練中/結束自動存
        save_on_each_node=False,
        load_best_model_at_end=False,
        gradient_checkpointing=False, 
    )

    from transformers import Trainer

    trainer = Trainer(
        model=model,
        args=train_args,                # 這裡包含 deepspeed=args.deepspeed
        train_dataset=tokenized,
        tokenizer=tokenizer,
        data_collator=collator,
    )

    # --- 訓練 ---
    train_out = trainer.train()
    print(train_out)
    metrics = trainer.state.log_history[-1]
    tokens_per_s = metrics["train_samples_per_second"] * 284.3  # 以實際 avg seq len 替代
    print(f"=== Approx. Token Throughput: {tokens_per_s:.2f} tokens/s ===")

    # --- 保存：A) LoRA adapter（最省空間） ---
    adapter_dir = os.path.join(args.output_dir, "adapter")
    os.makedirs(adapter_dir, exist_ok=True)
    trainer.model.save_pretrained(adapter_dir)
    tokenizer.save_pretrained(adapter_dir)
    print(f"[saved] LoRA adapter to: {adapter_dir}")

    # --- C) 不再使用 trainer.save_model() 避免 CPU OOM ---
    print(f"[saved] LoRA adapter only (no full model save to avoid CPU OOM)")

    # --- B) 可選：合併 LoRA → 輸出完整模型 ---
    if args.merge_and_save:
        from peft import AutoPeftModelForCausalLM
        merged_dir = os.path.join(args.output_dir, "merged")
        os.makedirs(merged_dir, exist_ok=True)

        merged = AutoPeftModelForCausalLM.from_pretrained(adapter_dir, torch_dtype=dtype or "auto").merge_and_unload()
        # 使用安全序列化（safetensors）
        merged.save_pretrained(merged_dir, safe_serialization=True)
        tokenizer.save_pretrained(merged_dir)
        print(f"[saved] merged full model to: {merged_dir}")

if __name__ == "__main__":
    main()
