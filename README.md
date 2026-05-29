# What Does LoRA Change? — Interpreting LoRA Fine-Tuning in LLMs

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mlahozy21/Interpreting-LoRA-Fine-Tuning/blob/main/notebooks/study.ipynb)

LoRA is the default way to fine-tune LLMs, yet it is rarely asked **what** a LoRA
adapter actually changes inside the model. This project fine-tunes `Qwen2.5-1.5B`
with LoRA and then **measures** the change with two training-free, post-hoc
diagnostics, across an ablation over the LoRA rank and the adapted modules.

A short paper-style write-up is in [`paper/report.pdf`](paper/report.pdf).

## Research questions

1. **Where** does LoRA put its capacity — which layers and which projections
   (`q/k/v/o` vs. the MLP `gate/up/down`) get the largest updates?
2. **How much** do the hidden representations move once the adapter is enabled,
   and at what depth?
3. How do (1)–(2) depend on the **rank** and the **set of adapted modules**?

## The two diagnostics

- **Effective update magnitude** — for every adapted weight `W`, form the explicit
  update `ΔW = (α/r)·B·A` and report the relative Frobenius norm `‖ΔW‖/‖W‖`
  (how much it changes) and the **effective rank** of `ΔW` (how many directions it
  uses). → *where* the capacity goes.
- **Representation drift** — run held-out text through the model with the adapter
  **on** vs **off** and compare the hidden states layer by layer (cosine similarity
  and relative L2). → *how much* and *where* behaviour is reshaped.

Both are implemented in `src/lora_interp/analysis.py` and are model-agnostic
(any LoRA-adapted 🤗 Transformer).

## Run

One click in Colab (badge above) or locally:

```bash
pip install -e .            # installs the lora_interp package
# (on Colab also: pip uninstall -y torchao)

python scripts/quick_demo.py                                   # one small config
python scripts/run_study.py --ranks 4 16 64 --targets attn all # full ablation
```

Outputs: per-config figures in `figures/` (update heatmap, update-vs-depth, drift
curve) and a `results.csv` summary across the ablation.

## Repository layout

```
.
├── src/lora_interp/
│   ├── train.py       # LoRA instruction fine-tuning (returns the in-memory model)
│   ├── analysis.py    # update-magnitude + representation-drift diagnostics
│   ├── plots.py       # figures
│   └── utils.py       # seeding, precision, model loading
├── scripts/
│   ├── run_study.py   # rank x module-set ablation -> figures + results.csv
│   └── quick_demo.py  # single-config end-to-end
├── notebooks/study.ipynb
├── paper/report.tex (+ report.pdf)
└── figures/           # generated plots
```

## License

Released under the MIT License — see `LICENSE`.
