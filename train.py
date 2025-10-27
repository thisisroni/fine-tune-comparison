#!/usr/bin/env python3
import argparse, os
from packaging import version

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, DataCollatorForLanguageModeling, Trainer
from peft import LoraConfig, get_peft_model
import trl

from itertools import islice
import numpy as np

def get_args():
    p = argparse.ArgumentParser(
        description="Fine-tune Large Language Models with LoRA/QLoRA",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic LoRA training
  python train.py --base_model NousResearch/Llama-2-7b-chat-hf --lora --fp16 --epochs 1.0
  
  # QLoRA training with custom parameters
  python train.py --base_model NousResearch/Llama-2-7b-chat-hf --qlora --fp16 --lora_r 32 --lr 2e-4
  
  # DeepSpeed training
  deepspeed train.py --base_model NousResearch/Llama-2-7b-chat-hf --deepspeed ds_config.json --lora
        """
    )
    
    # === Model and Dataset ===
    model_group = p.add_argument_group('Model and Dataset')
    model_group.add_argument("--base_model", type=str, default="NousResearch/Llama-2-7b-chat-hf", 
                            help="Base model path or HuggingFace model ID")
    model_group.add_argument("--dataset", type=str, default="mlabonne/guanaco-llama2-1k",
                            help="Dataset path or HuggingFace dataset ID")
    model_group.add_argument("--output_dir", type=str, default="./results_modified",
                            help="Directory to save training outputs (adapters, checkpoints)")
    
    # === Training Parameters ===
    train_group = p.add_argument_group('Training Parameters')
    train_group.add_argument("--epochs", type=float, default=1.0,
                            help="Number of training epochs (can be fractional)")
    train_group.add_argument("--per_device_bs", type=int, default=1,
                            help="Batch size per device")
    train_group.add_argument("--grad_accum", type=int, default=1,
                            help="Gradient accumulation steps")
    train_group.add_argument("--lr", type=float, default=1e-3,
                            help="Learning rate")
    train_group.add_argument("--weight_decay", type=float, default=0.0,
                            help="Weight decay for regularization")
    train_group.add_argument("--warmup_steps", type=int, default=0, 
                            help="Number of warmup steps for learning rate scheduler")
    train_group.add_argument("--warmup_ratio", type=float, default=0.0, 
                            help="Warmup ratio of total training steps (alternative to warmup_steps)")
    
    # === Data Processing ===
    data_group = p.add_argument_group('Data Processing')
    data_group.add_argument("--max_length", type=int, default=350,
                           help="Maximum sequence length for input text")
    data_group.add_argument("--packing", action="store_true",
                           help="Enable sequence packing for efficiency")
    
    # === Logging and Saving ===
    log_group = p.add_argument_group('Logging and Saving')
    log_group.add_argument("--logging_steps", type=int, default=20,
                          help="Log training metrics every N steps")
    log_group.add_argument("--save_steps", type=int, default=200,
                          help="Save checkpoint every N steps")
    log_group.add_argument("--merge_and_save", action="store_true",
                          help="Merge LoRA weights with base model and save full model")
    
    # === Precision and Optimization ===
    precision_group = p.add_argument_group('Precision and Optimization')
    precision_group.add_argument("--bf16", action="store_true",
                                help="Use bfloat16 mixed precision (requires Ampere+ GPUs)")
    precision_group.add_argument("--fp16", action="store_true",
                                help="Use float16 mixed precision")
    precision_group.add_argument("--use_flash_attn2", action="store_true",
                                help="Use Flash Attention 2 for faster training")
    
    # === Fine-tuning Methods ===
    finetune_group = p.add_argument_group('Fine-tuning Methods')
    finetune_group.add_argument("--qlora", action="store_true",
                               help="Enable QLoRA (4-bit quantized LoRA)")
    finetune_group.add_argument("--seed", type=int, default=42,
                               help="Random seed for reproducibility")

    # === DeepSpeed Integration ===
    deepspeed_group = p.add_argument_group('DeepSpeed Integration')
    deepspeed_group.add_argument("--deepspeed", type=str, default=None, 
                                help="Path to DeepSpeed configuration JSON file")
    deepspeed_group.add_argument("--local_rank", type=int, default=-1, 
                                help="Local rank for distributed training (used by DeepSpeed/torchrun)")
    
    # === Alternative Parameter Names (for compatibility) ===
    alias_group = p.add_argument_group('Alternative Parameter Names')
    alias_group.add_argument("--model_name_or_path", type=str, 
                            help="Alternative name for --base_model")
    alias_group.add_argument("--per_device_train_batch_size", type=int, 
                            help="Alternative name for --per_device_bs")
    alias_group.add_argument("--gradient_accumulation_steps", type=int, 
                            help="Alternative name for --grad_accum")
    alias_group.add_argument("--num_train_epochs", type=float, 
                            help="Alternative name for --epochs")
    alias_group.add_argument("--learning_rate", type=float, 
                            help="Alternative name for --lr")
    alias_group.add_argument("--max_seq_length", type=int, 
                            help="Alternative name for --max_length")

    # === LoRA Configuration ===
    lora_group = p.add_argument_group('LoRA Configuration')
    lora_group.add_argument("--lora", action="store_true",
                           help="Enable LoRA (Low-Rank Adaptation) fine-tuning")
    lora_group.add_argument("--lora_r", type=int, default=64,
                           help="LoRA rank (dimension of adaptation matrices)")
    lora_group.add_argument("--lora_alpha", type=int, default=16,
                           help="LoRA alpha parameter (scaling factor)")
    lora_group.add_argument("--lora_dropout", type=float, default=0.05,
                           help="Dropout rate for LoRA layers")

    # === Attention Implementation ===
    attn_group = p.add_argument_group('Attention Implementation')
    attn_group.add_argument("--attn_impl", type=str, default="eager",
                           choices=["eager", "flash_attn2"],
                           help="Attention implementation: 'eager' (default) or 'flash_attn2' (faster)")

    args = p.parse_args()

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

    # 設定 tokenizer 
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

    # DeepSpeed
    from_kwargs = dict(
        low_cpu_mem_usage=True,
        torch_dtype=(dtype if dtype is not None else "auto"),
        quantization_config=bnb_config,     # QLoRA 
    )
    
    if args.deepspeed is None:
        from_kwargs["device_map"] = "auto"
    else:
        from_kwargs["device_map"] = {"": 0}

    model = AutoModelForCausalLM.from_pretrained(args.base_model, **from_kwargs)
    model.config.use_cache = False

    # FlashAttention
    if args.attn_impl == "flash_attn2":
        try:
            model.config.attn_implementation = "flash_attention_2"
        except Exception:
            pass

    # Dataset 
    train_ds = load_dataset(args.dataset, split="train")

    def estimate_avg_len(ds, tok, max_len=350, sample=256):
        lens = []
        for ex in islice(ds, 0, sample):
            ids = tok(ex["text"], max_length=max_len, truncation=True, add_special_tokens=True)["input_ids"]
            lens.append(len(ids))
        return float(np.mean(lens))

    avg_len = estimate_avg_len(train_ds, tokenizer, args.max_length, 256)
    print(f"[INFO] estimated avg seq len = {avg_len:.1f}")

    # LoRA 
    peft_cfg = None
    if args.lora or args.qlora:
        peft_cfg = LoraConfig(
            r=args.lora_r,                  
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

    if hasattr(model, "print_trainable_parameters"):
        model.print_trainable_parameters()


    def to_text(example):
        if "text" in example and isinstance(example["text"], str):
            return example["text"]
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

    tokenized = train_ds.map(tokenize_fn, remove_columns=train_ds.column_names, batched=False)

    collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

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
        deepspeed=args.deepspeed,    
        report_to=[],
        save_strategy="no",           
        save_on_each_node=False,
        load_best_model_at_end=False,
        gradient_checkpointing=False, 
    )


    trainer = Trainer(
        model=model,
        args=train_args,               
        train_dataset=tokenized,
        tokenizer=tokenizer,
        data_collator=collator,
    )

    train_out = trainer.train()
    print(train_out)
    metrics = trainer.state.log_history[-1]
    tokens_per_s = metrics["train_samples_per_second"] * 284.3 
    print(f"=== Approx. Token Throughput: {tokens_per_s:.2f} tokens/s ===")

    adapter_dir = os.path.join(args.output_dir, "adapter")
    os.makedirs(adapter_dir, exist_ok=True)
    trainer.model.save_pretrained(adapter_dir)
    tokenizer.save_pretrained(adapter_dir)
    print(f"[saved] LoRA adapter to: {adapter_dir}")

    print(f"[saved] LoRA adapter only (no full model save to avoid CPU OOM)")

    if args.merge_and_save:
        from peft import AutoPeftModelForCausalLM
        merged_dir = os.path.join(args.output_dir, "merged")
        os.makedirs(merged_dir, exist_ok=True)

        merged = AutoPeftModelForCausalLM.from_pretrained(adapter_dir, torch_dtype=dtype or "auto").merge_and_unload()
        merged.save_pretrained(merged_dir, safe_serialization=True)
        tokenizer.save_pretrained(merged_dir)
        print(f"[saved] merged full model to: {merged_dir}")

if __name__ == "__main__":
    main()
