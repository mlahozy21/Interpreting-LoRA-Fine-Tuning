# What Does LoRA Change? — Interpreting LoRA Fine-Tuning in LLMs

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mlahozy21/Interpreting-LoRA-Fine-Tuning/blob/main/notebooks/study.ipynb)

> **TL;DR — LoRA makes tiny (≈1%), high-rank, attention-biased edits that *rescale* rather than *rotate* representations.**

LoRA is the default way to fine-tune LLMs, yet it is rarely asked **what** a LoRA
adapter actually changes inside the model. This project fine-tunes `Qwen2.5-1.5B`
with LoRA and then **measures** the change with two training-free, post-hoc
diagnostics, across an ablation over the LoRA rank and the adapted modules.

A short report is in [`paper/report.pdf`](paper/report.pdf).

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

## Results

Fine-tuned **Qwen2.5-1.5B** on 2,000 Alpaca examples per configuration (1 epoch),
ablating the LoRA rank `r ∈ {4, 16, 64}` and the adapted module set
(`attn` = q/k/v/o; `all` = + gate/up/down). Training loss is near-flat across
configs (1.39–1.47; best: rank 64 / all, 1.41) — already a hint of diminishing returns.

| rank | modules | mean ρ = ‖ΔW‖/‖W‖ | ρ (attn) | ρ (mlp) | e-rank(ΔW) | drift (rel. L2) |
|----:|:-------|:----:|:----:|:----:|:----:|:----:|
| 4  | attn | 0.0063 | 0.0063 | —      | 3.24  | 0.299 |
| 16 | attn | 0.0088 | 0.0088 | —      | 11.77 | 0.320 |
| 64 | attn | 0.0145 | 0.0145 | —      | 44.90 | 0.314 |
| 4  | all  | 0.0039 | 0.0042 | 0.0035 | 3.55  | 0.250 |
| 16 | all  | 0.0063 | 0.0069 | 0.0056 | 13.41 | 0.257 |
| 64 | all  | 0.0117 | 0.0127 | 0.0104 | 52.01 | 0.224 |

**Four findings:**

1. **Updates are tiny in magnitude.** ρ stays at ~0.4%–1.5%: LoRA makes small, targeted
   edits to the weights rather than large changes.
2. **LoRA uses (almost) its full rank budget.** The effective rank of ΔW tracks `r` closely
   (3.2 at r=4, 11.8 at r=16, ~45–52 at r=64): the adaptation is genuinely high-dimensional,
   not a rank-1 shortcut.
3. **Attention changes more than the MLP.** Under `all`, attention projections consistently
   receive larger relative updates than the MLP at every rank.
4. **Fine-tuning rescales rather than rotates representations.** Final-layer cosine
   similarity between adapter-on/-off stays high (0.96–0.98) while relative-L2 drift is
   sizeable (0.22–0.32): direction is largely preserved, magnitude/offset moves. Spreading
   capacity across all modules yields lower per-layer drift than attention-only at matched rank.

Together with the near-flat training loss, this suggests that for this model/task a small
rank already captures most of the adaptation, and the extra capacity of large ranks is only
partly used.

*Limitations:* single model, single dataset, one epoch — exploratory, not conclusive.
A second model (and DoRA/other PEFT variants) would be needed to claim generality.


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
