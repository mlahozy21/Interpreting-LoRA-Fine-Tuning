"""Two interpretability probes of a LoRA-fine-tuned model.

1) `adapter_update_norms` — for every LoRA-adapted weight W, how large is the
   effective update  dW = (alpha/r) * B @ A  relative to W (Frobenius), and what
   is its singular-value spectrum (effective rank)? This shows *where* (which
   layers / projection types) LoRA puts its capacity.

2) `representation_drift` — feed held-out text through the model with the LoRA
   adapter ENABLED vs DISABLED and compare the hidden states layer by layer
   (cosine similarity and relative L2). This shows *how much* the internal
   representations actually move, and at which depth.
"""

from __future__ import annotations

import re

import numpy as np
import torch


# --------------------------------------------------------------------------- #
# 1) Effective update magnitude per adapted module
# --------------------------------------------------------------------------- #
def _iter_lora_modules(model):
    for name, module in model.named_modules():
        if hasattr(module, "lora_A") and hasattr(module, "lora_B") \
                and hasattr(module, "base_layer"):
            keys = list(module.lora_A.keys())
            if keys:
                yield name, module, keys[0]


def _layer_index(name: str):
    m = re.search(r"\.layers\.(\d+)\.", name)
    return int(m.group(1)) if m else -1


def _module_type(name: str) -> str:
    return name.split(".")[-1]  # e.g. q_proj, gate_proj, ...


@torch.no_grad()
def adapter_update_norms(model) -> list[dict]:
    """Per-adapted-module relative update norm and effective rank of dW."""
    rows = []
    for name, module, adapter in _iter_lora_modules(model):
        A = module.lora_A[adapter].weight.float()      # (r, in)
        B = module.lora_B[adapter].weight.float()      # (out, r)
        scaling = float(module.scaling[adapter])
        dW = scaling * (B @ A)                          # (out, in)
        W = module.base_layer.weight.float()
        dnorm = torch.linalg.norm(dW).item()
        wnorm = torch.linalg.norm(W).item()
        # effective rank of dW from its singular values (participation ratio)
        sv = torch.linalg.svdvals(dW)
        p = (sv / sv.sum()).clamp_min(1e-12)
        eff_rank = float(torch.exp(-(p * p.log()).sum()))
        rows.append({
            "module": name,
            "layer": _layer_index(name),
            "type": _module_type(name),
            "rel_update": dnorm / (wnorm + 1e-12),
            "update_norm": dnorm,
            "weight_norm": wnorm,
            "effective_rank": eff_rank,
        })
    return rows


# --------------------------------------------------------------------------- #
# 2) Representation drift (adapter on vs off)
# --------------------------------------------------------------------------- #
@torch.no_grad()
def _hidden_states(model, tokenizer, texts, max_len=256):
    """Mean-pooled hidden states per layer, shape (n_layers+1, n_texts, hidden)."""
    pooled_per_text = []
    device = next(model.parameters()).device
    for t in texts:
        enc = tokenizer(t, return_tensors="pt", truncation=True, max_length=max_len).to(device)
        out = model(**enc, output_hidden_states=True)
        mask = enc["attention_mask"].unsqueeze(-1).float()      # (1, T, 1)
        pooled = [((h * mask).sum(1) / mask.sum(1)).squeeze(0).float().cpu()
                  for h in out.hidden_states]                   # list of (hidden,)
        pooled_per_text.append(torch.stack(pooled))             # (n_layers+1, hidden)
    return torch.stack(pooled_per_text, dim=1)                  # (n_layers+1, n_texts, hidden)


@torch.no_grad()
def representation_drift(model, tokenizer, texts, max_len=256) -> dict:
    """Per-layer drift between adapter-enabled and adapter-disabled hidden states."""
    model.eval()
    h_on = _hidden_states(model, tokenizer, texts, max_len)
    with model.disable_adapter():
        h_off = _hidden_states(model, tokenizer, texts, max_len)

    cos = torch.nn.functional.cosine_similarity(h_on, h_off, dim=-1).mean(dim=1)  # (n_layers+1,)
    rel_l2 = (torch.linalg.norm(h_on - h_off, dim=-1)
              / (torch.linalg.norm(h_off, dim=-1) + 1e-12)).mean(dim=1)
    return {
        "layer": list(range(h_on.shape[0])),
        "cosine_similarity": cos.tolist(),
        "relative_l2": rel_l2.tolist(),
    }
