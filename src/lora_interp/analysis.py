"""Two interpretability probes of a LoRA-fine-tuned model.

1a) `adapter_update_norms` -- DIRECTIONAL update only: for every LoRA-adapted
    weight W, form  dW = (alpha/r) * B @ A  and report its relative Frobenius
    norm and participation ratio. This is exact for plain LoRA but, for DoRA, it
    ignores the learned magnitude vector and therefore CANNOT distinguish DoRA
    from LoRA.

1b) `exact_update_norms` -- EXACT effective update: dW = (merged effective
    weight) - (base weight), computed per-module in fp32 from the stored
    adapter tensors. For plain LoRA this equals (alpha/r)*B@A; for DoRA it also
    applies the magnitude rescaling  m * (W + dV)/||W + dV||_col - W, where dV is
    the directional update and m is `lora_magnitude_vector`. This backs the
    "DoRA vs LoRA" comparison.

2) `representation_drift` -- feed held-out text through the model with the LoRA
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


def participation_ratio(delta: torch.Tensor) -> float:
    """Participation ratio of a matrix's singular-value spectrum.

    Definition: with normalized singular values p_i = s_i / sum_j s_j, the
    participation ratio is exp(-sum_i p_i * log p_i) -- the exponential of the
    Shannon entropy of the (L1-)normalized spectrum. It is *not* the matrix
    rank: it is a soft "how many singular directions are effectively used"
    measure. It lies in [1, min(out, in)] and equals k exactly when k singular
    values are equal and the rest are zero (so a rank-r update has participation
    ratio <= r, with equality only for a flat spectrum).
    """
    sv = torch.linalg.svdvals(delta.float())
    p = (sv / sv.sum()).clamp_min(1e-12)
    return float(torch.exp(-(p * p.log()).sum()))


# Backwards-compatible alias: the original code/README called this the
# "effective rank". It is really the participation ratio (see above); the name
# is kept so existing callers and saved JSON keys keep working.
effective_rank = participation_ratio


def _directional_delta(module, adapter) -> torch.Tensor:
    """The directional LoRA update  dV = (alpha/r) * B @ A  in fp32 (out, in)."""
    A = module.lora_A[adapter].weight.float()      # (r, in)
    B = module.lora_B[adapter].weight.float()      # (out, r)
    scaling = float(module.scaling[adapter])
    return scaling * (B @ A)                        # (out, in)


def _dora_magnitude(module, adapter):
    """Return the DoRA magnitude vector m (shape (out,)) in fp32, or None.

    In PEFT, DoRA stores `lora_magnitude_vector` as a ModuleDict / ParameterDict
    whose entries expose the learned per-output magnitude either as `.weight` (a
    Parameter) or are themselves a Parameter. We handle both and otherwise
    return None (plain LoRA, no magnitude component).
    """
    mag = getattr(module, "lora_magnitude_vector", None)
    if mag is None:
        return None
    try:
        entry = mag[adapter]
    except (KeyError, TypeError):
        return None
    if entry is None:
        return None
    weight = getattr(entry, "weight", entry)
    if not torch.is_tensor(weight):
        return None
    return weight.detach().float().reshape(-1)      # (out,)


def _exact_delta(module, adapter) -> torch.Tensor:
    """The EXACT effective update  dW = W_eff - W0  in fp32 (out, in).

    Plain LoRA:  W_eff = W0 + dV, so dW = dV.
    DoRA:        W_eff = m * (W0 + dV) / col_norm(W0 + dV), where col_norm is the
                 per-output-row L2 norm, so the magnitude rescaling is included
                 and dW differs from dV in general.
    """
    W = module.base_layer.weight.float()           # (out, in)
    dV = _directional_delta(module, adapter)        # (out, in)
    m = _dora_magnitude(module, adapter)
    if m is None:
        return dV                                   # plain LoRA: exact == directional
    numerator = W + dV                              # (out, in)
    # PEFT computes the norm per output row (dim=1), keeping (out, 1).
    col_norm = torch.linalg.norm(numerator, dim=1, keepdim=True).clamp_min(1e-12)
    W_eff = (m.reshape(-1, 1) / col_norm) * numerator
    return W_eff - W


def _norm_rows(model, delta_fn):
    rows = []
    for name, module, adapter in _iter_lora_modules(model):
        dW = delta_fn(module, adapter)                  # (out, in)
        W = module.base_layer.weight.float()
        dnorm = torch.linalg.norm(dW).item()
        wnorm = torch.linalg.norm(W).item()
        pr = participation_ratio(dW)
        rows.append({
            "module": name,
            "layer": _layer_index(name),
            "type": _module_type(name),
            "rel_update": dnorm / (wnorm + 1e-12),
            "update_norm": dnorm,
            "weight_norm": wnorm,
            "participation_ratio": pr,
            "effective_rank": pr,   # alias kept for backward compatibility
        })
    return rows


@torch.no_grad()
def adapter_update_norms(model) -> list[dict]:
    """DIRECTIONAL per-module update norm + participation ratio of dV=(alpha/r)B@A.

    Exact for plain LoRA, but ignores DoRA's magnitude vector -- use
    `exact_update_norms` when DoRA may be present.
    """
    return _norm_rows(model, _directional_delta)


@torch.no_grad()
def exact_update_norms(model) -> list[dict]:
    """EXACT per-module effective update norm + participation ratio.

    Computes dW = W_eff - W0 per module in fp32 from the stored adapter tensors,
    including DoRA's magnitude rescaling when a `lora_magnitude_vector` is
    present. For plain LoRA this is identical to `adapter_update_norms`. This is
    the diagnostic that can genuinely distinguish DoRA from LoRA.
    """
    return _norm_rows(model, _exact_delta)


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
        # Masked mean-pool. We tokenize one text at a time, so there is no
        # padding here and attention_mask is all-ones; the mask is applied
        # anyway so this helper stays correct if a *padded* batch is ever passed
        # in (e.g. multiple texts tokenized together with padding=True).
        mask = enc["attention_mask"].unsqueeze(-1).float()      # (B, T, 1)
        pooled = [((h * mask).sum(1) / mask.sum(1).clamp_min(1e-12)).float().cpu()
                  for h in out.hidden_states]                   # list of (B, hidden)
        # B == 1 here; keep the batch dim and concatenate across texts below.
        pooled_per_text.append(torch.stack(pooled))             # (n_layers+1, B, hidden)
    return torch.cat(pooled_per_text, dim=1)                    # (n_layers+1, n_texts, hidden)


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
