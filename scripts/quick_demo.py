"""Minimal single-config run: train one LoRA adapter and print/save the analyses."""

import json

from lora_interp.analysis import adapter_update_norms, representation_drift
from lora_interp.plots import plot_drift, plot_update_heatmap
from lora_interp.train import train_lora

HELDOUT = [
    "### Instruction:\nExplain what overfitting is.\n\n### Response:\n",
    "### Instruction:\nWrite a Python function that reverses a string.\n\n### Response:\n",
]

if __name__ == "__main__":
    model, tok = train_lora(rank=16, target="all", max_samples=500, epochs=1)
    rows = adapter_update_norms(model)
    print("Top-5 most-updated modules (||dW||/||W||):")
    for r in sorted(rows, key=lambda x: -x["rel_update"])[:5]:
        print(f"  {r['module']:45s} {r['rel_update']:.3f}  eff_rank={r['effective_rank']:.1f}")
    plot_update_heatmap(rows, "figures/demo_update_heatmap.png")
    drift = representation_drift(model, tok, HELDOUT)
    plot_drift(drift, "figures/demo_drift.png")
    json.dump({"update": rows, "drift": drift}, open("figures/demo_results.json", "w"), indent=1)
    print("Figures saved under figures/.")
