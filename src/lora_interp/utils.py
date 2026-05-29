"""Shared helpers: seeding, precision, model/tokenizer loading."""

from __future__ import annotations

import random

import numpy as np
import torch


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def best_dtype():
    """bf16 on capable GPUs (A100/L4), else fp16, else fp32 (CPU)."""
    if torch.cuda.is_available():
        return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    return torch.float32


def load_base(model_name: str):
    """Load the base causal LM and its tokenizer."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model_name)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_name, dtype=best_dtype(),
        device_map="auto" if torch.cuda.is_available() else None,
    )
    return model, tok
