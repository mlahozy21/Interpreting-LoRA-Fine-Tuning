"""CPU tests of the analysis probes on a synthetic LoRA-adapted module.

No model download: we build a fake PEFT-style module (base_layer + lora_A/B +
scaling) and check that `adapter_update_norms` computes exactly what it claims.
"""
import sys

import torch
import torch.nn as nn

sys.path.insert(0, "src")

from lora_interp.analysis import _layer_index, _module_type, adapter_update_norms  # noqa: E402


class FakeLoRALinear(nn.Module):
    """Mimics peft.tuners.lora.Linear: base_layer + lora_A/lora_B ModuleDicts."""

    def __init__(self, d_in, d_out, r, scaling):
        super().__init__()
        self.base_layer = nn.Linear(d_in, d_out, bias=False)
        self.lora_A = nn.ModuleDict({"default": nn.Linear(d_in, r, bias=False)})
        self.lora_B = nn.ModuleDict({"default": nn.Linear(r, d_out, bias=False)})
        self.scaling = {"default": scaling}


def make_model(r=2, scaling=0.5):
    torch.manual_seed(0)
    root = nn.Module()           # named paths must look like "model.layers.3..."
    inner = nn.Module()
    layers = nn.ModuleList([nn.Module() for _ in range(4)])
    attn = nn.Module()
    attn.q_proj = FakeLoRALinear(8, 8, r=r, scaling=scaling)
    layers[3].self_attn = attn
    inner.layers = layers
    root.model = inner
    return root, attn.q_proj


def test_name_parsing():
    assert _layer_index("model.layers.7.self_attn.q_proj") == 7
    assert _layer_index("lm_head") == -1
    assert _module_type("model.layers.7.mlp.gate_proj") == "gate_proj"


def test_update_norm_matches_manual_computation():
    model, mod = make_model(r=2, scaling=0.5)
    rows = adapter_update_norms(model)
    assert len(rows) == 1
    row = rows[0]
    A = mod.lora_A["default"].weight.float()
    B = mod.lora_B["default"].weight.float()
    dW = 0.5 * (B @ A)
    expected_rel = torch.linalg.norm(dW) / torch.linalg.norm(mod.base_layer.weight)
    assert abs(row["rel_update"] - expected_rel.item()) < 1e-6
    assert row["layer"] == 3 and row["type"] == "q_proj"


def test_effective_rank_of_rank1_update_is_one():
    model, _ = make_model(r=1, scaling=1.0)
    row = adapter_update_norms(model)[0]
    assert abs(row["effective_rank"] - 1.0) < 1e-3


def test_effective_rank_bounded_by_r():
    model, _ = make_model(r=4, scaling=1.0)
    row = adapter_update_norms(model)[0]
    assert 1.0 <= row["effective_rank"] <= 4.0 + 1e-6
