# What Does LoRA Change? — Interpreting LoRA Fine-Tuning in LLMs

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mlahozy21/Interpreting-LoRA-Fine-Tuning/blob/main/notebooks/study.ipynb)

> **TL;DR — LoRA makes tiny (≲1%), high-rank, attention-biased edits that *rescale* rather than *rotate* representations — replicated on a second model family. A DoRA-vs-LoRA comparison is included; the diagnostic that backs it (an *exact* merge-and-diff that reads DoRA's magnitude vector) has been corrected and the comparison must be re-run on GPU before any conclusion is drawn.**

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

- **Effective update magnitude** — for every adapted weight `W`, form the
  effective update `ΔW` and report the relative Frobenius norm `‖ΔW‖/‖W‖` (how
  much it changes) and the **participation ratio** of `ΔW` (a soft count of how
  many singular directions it uses; *not* the matrix rank — see the docstring of
  `participation_ratio`). Two estimators are provided: `adapter_update_norms`
  computes the *directional* update `ΔV = (α/r)·B·A` (exact for plain LoRA), and
  `exact_update_norms` computes the *exact* merged update `ΔW = W_eff − W₀`
  per-module in fp32 — for DoRA this includes the learned magnitude rescaling, so
  it is the estimator that can actually distinguish DoRA from LoRA. → *where* the
  capacity goes.
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

> **Reproducibility note:** the 6-row table below is *not* backed by a committed
> artifact — only `results_extension.csv` is in the repo. The full ablation needs
> Qwen2.5-1.5B on GPU and cannot be regenerated on CPU here. Run
> `python scripts/run_study.py --ranks 4 16 64 --targets attn all` to regenerate
> it; that script writes `results.csv` to the repo root (commit it alongside the
> table). The numbers below are the reported values from the original GPU run.

| rank | modules | mean ρ = ‖ΔW‖/‖W‖ | ρ (attn) | ρ (mlp) | e-rank(ΔW) | drift (rel. L2) |
|----:|:-------|:----:|:----:|:----:|:----:|:----:|
| 4  | attn | 0.0063 | 0.0063 | —      | 3.24  | 0.299 |  <!-- e-rank column = participation ratio of ΔW -->
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

*Limitations of the original ablation:* single model, single dataset, one epoch.
The extension below addresses the model axis directly.

### Extension: does it generalise? (2nd model family + DoRA)

`notebooks/extension_dora_second_model.ipynb`
([Colab](https://colab.research.google.com/github/mlahozy21/Interpreting-LoRA-Fine-Tuning/blob/main/notebooks/extension_dora_second_model.ipynb))
re-runs the diagnostics on a **second model family** (SmolLM2-1.7B, Llama-style) and a
**second PEFT variant** (**DoRA**), with two methodological upgrades over the original
study: *exact* update norms (`exact_update_norms`: per-module `ΔW = W_eff − W₀` in
fp32, reading DoRA's magnitude vector so its magnitude component is actually
included), and a **behavioural validation** — held-out instruction loss, base vs
fine-tuned — so every configuration is checked to have actually adapted before its
weights are interpreted.

> **Correction (important):** the DoRA rows in the table below were produced by an
> earlier version of the diagnostic that used the *directional-only* formula
> `ΔW = (α/r)·B·A` and never read DoRA's magnitude vector — so by construction it
> could not tell DoRA apart from LoRA. The exact merge-and-diff diagnostic is now
> implemented (`src/lora_interp/analysis.py::exact_update_norms`, unit-tested to
> differ from the directional one on a synthetic DoRA module), and `run_study.py`
> now uses it. **The DoRA-vs-LoRA comparison must be regenerated on GPU** with the
> corrected diagnostic before any conclusion is drawn; the numbers shown are kept
> only as the (now-superseded) record of the original run.

Rank 16, all modules, 2,000 Alpaca examples, 1 epoch (`results_extension.csv`):

| model | variant | held-out loss (base → ft) | mean ρ = ‖ΔW‖/‖W‖ | ρ (attn) | ρ (mlp) | e-rank(ΔW) | drift cos | drift rel-L2 |
|:--|:--|:--:|:--:|:--:|:--:|:--:|:--:|:--:|
| Qwen2.5-1.5B | LoRA | 1.976 → 1.454 | 0.0063 | 0.0068 | 0.0056 | 13.4 | 0.971 | 0.256 |
| Qwen2.5-1.5B | DoRA | 1.976 → 1.454 | 0.0062 | 0.0068 | 0.0056 | 13.5 | 0.971 | 0.259 |
| SmolLM2-1.7B | LoRA | 2.229 → 1.374 | 0.0022 | 0.0024 | 0.0020 | 10.2 | 0.926 | 0.377 |
| SmolLM2-1.7B | DoRA | 2.229 → 1.371 | 0.0022 | 0.0023 | 0.0020 | 10.2 | 0.928 | 0.373 |

Every configuration adapts behaviourally (held-out loss drops ~0.5–0.9 nats), so the
weight-space diagnostics are interpreting *successful* fine-tuning runs. The four
original findings replicate on the second model family:

1. **Tiny updates** — mean ρ stays ≤ 0.6% (SmolLM2 even smaller, ~0.2%).
2. **High effective rank** — e-rank(ΔW) ≈ 13.4/16 on Qwen and ≈ 10.2/16 on SmolLM2:
   most of the rank budget is used, with a model-dependent degree.
3. **Attention bias** — ρ(attn) > ρ(mlp) in all four configurations.
4. **Rescale, not rotate** — final-layer cosine stays high (0.97 / 0.93) with sizeable
   relative-L2 drift (0.26 / 0.38); SmolLM2 moves direction somewhat more than Qwen.

**DoRA vs LoRA (to be re-evaluated):** the original run reported the two as
near-identical, but that comparison rested on the directional-only diagnostic that
ignored DoRA's magnitude vector, so it was circular and is *not* a valid conclusion.
With the corrected exact diagnostic now in place, whether DoRA's magnitude
decomposition meaningfully changes the merged update at this scale and task is an
**open question to be re-evaluated on GPU** (`python scripts/run_study.py`). We do
not claim "indistinguishable".

*Remaining limitations:* one dataset, one epoch, rank 16 only in the extension.


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
├── notebooks/
│   ├── study.ipynb                       # original study, one click in Colab
│   └── extension_dora_second_model.ipynb # generality: 2nd model + DoRA + behavioural eval
├── tests/             # CPU tests of the probes on a synthetic LoRA module (CI)
├── paper/report.tex (+ report.pdf)
├── results_extension.csv  # generality study (2 models x LoRA/DoRA)
└── figures/           # generated plots
```

## License

Released under the MIT License — see `LICENSE`.
