"""Run the full LoRA-interpretability study (ablation over rank x module set).

For every configuration it: fine-tunes LoRA adapters, measures the per-module
update magnitude and the representation drift, saves the figures, and appends a
summary row to results.csv.

    python scripts/run_study.py --ranks 4 16 64 --targets attn all --max-samples 2000
Quick check:
    python scripts/run_study.py --ranks 16 --targets all --max-samples 500 --epochs 1
"""

import argparse
import csv
import gc
import json
import os

import torch

from lora_interp.analysis import adapter_update_norms, representation_drift
from lora_interp.plots import plot_drift, plot_update_by_layer, plot_update_heatmap
from lora_interp.train import train_lora

# Held-out instructions used to measure representation drift.
HELDOUT = [
    "### Instruction:\nExplain what overfitting is.\n\n### Response:\n",
    "### Instruction:\nSummarise the causes of the French Revolution.\n\n### Response:\n",
    "### Instruction:\nWrite a Python function that reverses a string.\n\n### Response:\n",
    "### Instruction:\nGive three tips for sleeping better.\n\n### Response:\n",
    "### Instruction:\nWhat is the capital of Australia?\n\n### Response:\n",
]


def main():
    ap = argparse.ArgumentParser(description="LoRA interpretability ablation study.")
    ap.add_argument("--model", default="Qwen/Qwen2.5-1.5B")
    ap.add_argument("--ranks", type=int, nargs="+", default=[4, 16, 64])
    ap.add_argument("--targets", nargs="+", default=["attn", "all"],
                    choices=["attn", "mlp", "all"])
    ap.add_argument("--max-samples", type=int, default=2000)
    ap.add_argument("--epochs", type=float, default=1.0)
    ap.add_argument("--fig-dir", default="figures")
    ap.add_argument("--results", default="results.csv")
    args = ap.parse_args()

    os.makedirs(args.fig_dir, exist_ok=True)
    summary = []

    for target in args.targets:
        for rank in args.ranks:
            tag = f"r{rank}_{target}"
            print(f"\n{'='*60}\nConfig: rank={rank}, target={target}\n{'='*60}")
            model, tok = train_lora(model_name=args.model, rank=rank, target=target,
                                    max_samples=args.max_samples, epochs=args.epochs,
                                    output_dir=f"outputs/{tag}")

            rows = adapter_update_norms(model)
            plot_update_heatmap(rows, f"{args.fig_dir}/update_heatmap_{tag}.png",
                                title=f"||dW||/||W||  (rank={rank}, {target})")
            plot_update_by_layer(rows, f"{args.fig_dir}/update_by_layer_{tag}.png",
                                 title=f"Update vs depth (rank={rank}, {target})")
            json.dump(rows, open(f"{args.fig_dir}/update_{tag}.json", "w"), indent=1)

            drift = representation_drift(model, tok, HELDOUT)
            plot_drift(drift, f"{args.fig_dir}/drift_{tag}.png",
                       title=f"Representation drift (rank={rank}, {target})")
            json.dump(drift, open(f"{args.fig_dir}/drift_{tag}.json", "w"), indent=1)

            attn_types = {"q_proj", "k_proj", "v_proj", "o_proj"}
            attn_u = [r["rel_update"] for r in rows if r["type"] in attn_types]
            mlp_u = [r["rel_update"] for r in rows if r["type"] not in attn_types]
            summary.append({
                "rank": rank, "target": target,
                "mean_rel_update": round(sum(r["rel_update"] for r in rows) / len(rows), 4),
                "mean_rel_update_attn": round(sum(attn_u) / len(attn_u), 4) if attn_u else None,
                "mean_rel_update_mlp": round(sum(mlp_u) / len(mlp_u), 4) if mlp_u else None,
                "mean_eff_rank": round(sum(r["effective_rank"] for r in rows) / len(rows), 2),
                "drift_cos_final": round(drift["cosine_similarity"][-1], 4),
                "drift_relL2_final": round(drift["relative_l2"][-1], 4),
            })
            del model
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    with open(args.results, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(summary[0].keys()))
        w.writeheader(); w.writerows(summary)
    print(f"\nSummary written to {args.results} and figures to {args.fig_dir}/")
    for s in summary:
        print(s)


if __name__ == "__main__":
    main()
