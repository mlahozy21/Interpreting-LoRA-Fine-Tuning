"""CPU tests of the analysis probes on a synthetic LoRA-adapted module.

No model download: we build a fake PEFT-style module (base_layer + lora_A/B +
scaling, plus an optional DoRA magnitude vector) and check that the update
diagnostics compute exactly what they claim.
"""
import sys

import torch
import torch.nn as nn

sys.path.insert(0, "src")

from lora_interp.analysis import (  # noqa: E402
    _layer_index,
    _module_type,
    adapter_update_norms,
    effective_rank,
    exact_update_norms,
    participation_ratio,
)


class FakeLoRALinear(nn.Module):
    """Mimics peft.tuners.lora.Linear: base_layer + lora_A/lora_B ModuleDicts.

    When `magnitude` is given it also exposes a `lora_magnitude_vector`
    (shape (d_out,)), mimicking a DoRA-adapted module.
    """

    def __init__(self, d_in, d_out, r, scaling, magnitude=None):
        super().__init__()
        self.base_layer = nn.Linear(d_in, d_out, bias=False)
        self.lora_A = nn.ModuleDict({"default": nn.Linear(d_in, r, bias=False)})
        self.lora_B = nn.ModuleDict({"default": nn.Linear(r, d_out, bias=False)})
        self.scaling = {"default": scaling}
        if magnitude is not None:
            self.lora_magnitude_vector = nn.ParameterDict(
                {"default": nn.Parameter(magnitude.clone())})


def make_model(r=2, scaling=0.5, magnitude=None):
    torch.manual_seed(0)
    root = nn.Module()           # named paths must look like "model.layers.3..."
    inner = nn.Module()
    layers = nn.ModuleList([nn.Module() for _ in range(4)])
    attn = nn.Module()
    attn.q_proj = FakeLoRALinear(8, 8, r=r, scaling=scaling, magnitude=magnitude)
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


def test_participation_ratio_of_rank1_update_is_one():
    model, _ = make_model(r=1, scaling=1.0)
    row = adapter_update_norms(model)[0]
    # participation_ratio is exposed under both keys (alias).
    assert abs(row["participation_ratio"] - 1.0) < 1e-3
    assert row["effective_rank"] == row["participation_ratio"]


def test_participation_ratio_bounded_by_r():
    """A rank-r update has participation ratio in [1, r] (<= r, not == rank)."""
    model, _ = make_model(r=4, scaling=1.0)
    row = adapter_update_norms(model)[0]
    assert 1.0 <= row["participation_ratio"] <= 4.0 + 1e-6


def test_participation_ratio_alias_and_definition():
    # The public alias points at the same function.
    assert effective_rank is participation_ratio
    # A flat rank-2 spectrum has participation ratio exactly 2.
    M = torch.zeros(4, 4)
    M[0, 0] = 3.0
    M[1, 1] = 3.0  # two equal singular values -> participation ratio == 2
    assert abs(participation_ratio(M) - 2.0) < 1e-5


def test_exact_equals_directional_for_plain_lora():
    """With no magnitude vector, the exact update == the directional update."""
    model, _ = make_model(r=2, scaling=0.5, magnitude=None)
    d_row = adapter_update_norms(model)[0]
    e_row = exact_update_norms(model)[0]
    assert abs(d_row["update_norm"] - e_row["update_norm"]) < 1e-6
    assert abs(d_row["rel_update"] - e_row["rel_update"]) < 1e-6


def test_exact_differs_from_directional_for_dora():
    """On a synthetic DoRA module the exact update must differ from directional.

    This locks in that the corrected diagnostic CAN distinguish DoRA from LoRA:
    the magnitude rescaling changes ||dW|| measurably.
    """
    torch.manual_seed(1)
    # A non-trivial magnitude vector (not equal to the column norms of W) forces
    # the DoRA rescaling to actually move the effective weight.
    magnitude = torch.linspace(0.5, 2.0, steps=8)
    model, mod = make_model(r=2, scaling=0.5, magnitude=magnitude)

    d_row = adapter_update_norms(model)[0]
    e_row = exact_update_norms(model)[0]

    # The two diagnostics must give materially different update norms.
    assert abs(d_row["update_norm"] - e_row["update_norm"]) > 1e-3
    rel_diff = abs(d_row["update_norm"] - e_row["update_norm"]) / e_row["update_norm"]
    assert rel_diff > 1e-3

    # Sanity-check the exact DoRA formula against a manual computation.
    W = mod.base_layer.weight.float()
    A = mod.lora_A["default"].weight.float()
    B = mod.lora_B["default"].weight.float()
    dV = 0.5 * (B @ A)
    num = W + dV
    col_norm = torch.linalg.norm(num, dim=1, keepdim=True)
    W_eff = (magnitude.reshape(-1, 1) / col_norm) * num
    dW_manual = W_eff - W
    assert abs(e_row["update_norm"] - torch.linalg.norm(dW_manual).item()) < 1e-4
