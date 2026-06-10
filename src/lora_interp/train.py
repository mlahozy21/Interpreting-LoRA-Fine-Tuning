"""LoRA instruction fine-tuning that returns the in-memory PEFT model.

Used by the study scripts so the adapter can be analysed right after training,
without reloading from disk.
"""

from __future__ import annotations

import torch
from datasets import load_dataset
from peft import LoraConfig, get_peft_model
from transformers import DataCollatorForLanguageModeling, Trainer, TrainingArguments

from .utils import best_dtype, load_base, set_seed

PROMPT_NO_INPUT = "### Instruction:\n{instruction}\n\n### Response:\n"
PROMPT_INPUT = "### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response:\n"

ATTN = ["q_proj", "k_proj", "v_proj", "o_proj"]
MLP = ["gate_proj", "up_proj", "down_proj"]
MODULE_SETS = {"attn": ATTN, "mlp": MLP, "all": ATTN + MLP}


def _format(ex):
    if ex.get("input", "").strip():
        p = PROMPT_INPUT.format(instruction=ex["instruction"], input=ex["input"])
    else:
        p = PROMPT_NO_INPUT.format(instruction=ex["instruction"])
    return p + ex["output"]


def train_lora(model_name="Qwen/Qwen2.5-1.5B", rank=16, target="all",
               dataset="tatsu-lab/alpaca", max_samples=2000, epochs=1.0,
               lr=2e-4, batch_size=8, grad_accum=2, max_len=512,
               output_dir=None, seed=42, use_dora=False):
    """Fine-tune LoRA (or DoRA, with `use_dora=True`) adapters and return
    (peft_model, tokenizer)."""
    set_seed(seed)
    model, tok = load_base(model_name)
    model.config.use_cache = False

    lora = LoraConfig(
        r=rank, lora_alpha=2 * rank, lora_dropout=0.05, bias="none",
        task_type="CAUSAL_LM", target_modules=MODULE_SETS[target],
        use_dora=use_dora,
    )
    model = get_peft_model(model, lora)
    model.print_trainable_parameters()

    ds = load_dataset(dataset, split="train")
    if max_samples and max_samples < len(ds):
        ds = ds.shuffle(seed=seed).select(range(max_samples))
    ds = ds.map(lambda e: tok(_format(e) + tok.eos_token, truncation=True, max_length=max_len),
                remove_columns=ds.column_names)

    use_bf16 = best_dtype() == torch.bfloat16
    args = TrainingArguments(
        output_dir=output_dir or f"outputs/r{rank}_{target}",
        num_train_epochs=epochs, per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=grad_accum, learning_rate=lr,
        lr_scheduler_type="cosine", warmup_ratio=0.03, logging_steps=25,
        save_strategy="no", bf16=use_bf16, fp16=not use_bf16 and torch.cuda.is_available(),
        report_to="none",
    )
    Trainer(model=model, args=args, train_dataset=ds,
            data_collator=