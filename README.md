# What Does LoRA Change? — Interpreting LoRA Fine-Tuning in LLMs

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mlahozy21/Interpreting-LoRA-Fine-Tuning/blob/main/notebooks/study.ipynb)

> **TL;DR — LoRA makes tiny (≲1%), high-rank, attention-biased edits that *rescale* rather than *rotate* representations — replicated on a second model family. DoRA vs LoRA: with an *exact* merge-and-diff diagnostic that reads DoRA's magnitude vector, DoRA produces nearly the same update magnitude and behavioural change as LoRA but spreads it across a far higher-rank effective update (participation ratio ≈ 540 / 1281 vs ≈ 13 / 10) — i.e. the magnitude component, not the direction, is what distinguishes DoRA.**

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

The extension below adds a second model family and a second PEFT variant.

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

> **Methodology — exact effective-update norms.** The `e-rank(ΔW)` column is the
> *participation ratio* of the **exact** merged update `ΔW = W_eff − W₀`
> (`src/lora_interp/analysis.py::exact_update_norms`), computed per-module in fp32
> from the stored adapter tensors. For plain LoRA this equals the directional update
> `(α/r)·B·A`; for DoRA it also **includes the learned magnitude rescaling**
> `m·(W₀+ΔV)/‖W₀+ΔV‖_row`, which is what makes the two variants distinguishable in
> weight space.

Rank 16, all modules, 2,000 Alpaca examples, 1 epoch (`results_extension.csv`):

| model | variant | held-out loss (base → ft) | mean ρ = ‖ΔW‖/‖W‖ | ρ (attn) | ρ (mlp) | e-rank(ΔW) | drift cos | drift rel-L2 |
|:--|:--|:--:|:--:|:--:|:--:|:--:|:--:|:--:|
| Qwen2.5-1.5B | LoRA | 1.974 → 1.454 | 0.0063 | 0.0069 | 0.0056 | 13.4 | 0.971 | 0.257 |
| Qwen2.5-1.5B | DoRA | 1.974 → 1.454 | 0.0066 | 0.0071 | 0.0060 | **540.5** | 0.971 | 0.259 |
| SmolLM2-1.7B | LoRA | 2.230 → 1.372 | 0.0022 | 0.0023 | 0.0020 | 10.2 | 0.927 | 0.375 |
| SmolLM2-1.7B | DoRA | 2.230 → 1.374 | 0.0028 | 0.0029 | 0.0027 | **1280.9** | 0.928 | 0.373 |

Every configuration adapts behaviourally (held-out loss drops ~0.5–0.9 nats), so the
weight-space diagnostics are interpreting *successful* fine-tuning runs. The first
three findings replicate on the second model family; the fourth is the new DoRA result:

1. **Tiny updates** — mean ρ stays ≤ 0.7% (SmolLM2 even smaller, ~0.2–0.3%), for both
   LoRA and DoRA. The magnitude of the change is essentially the same across variants.
2. **Attention bias** — ρ(attn) > ρ(mlp) in all four configurations.
3. **Rescale, not rotate** — final-layer cosine stays high (0.97 / 0.93) with sizeable
   relative-L2 drift (0.26 / 0.37); SmolLM2 moves direction somewhat more than Qwen.
   LoRA and DoRA drift almost identically — their *behavioural* effect is the same.
4. **DoRA differs from LoRA in the rank of the effective update, not its magnitude.**
   For LoRA the merged update is genuinely low-rank, tracking the budget `r`
   (participation ratio ≈ 13.4/16 on Qwen, ≈ 10.2/16 on SmolLM2). For DoRA the
   *exact* merged update is **high-rank** — ≈ 540 on Qwen and ≈ 1281 on SmolLM2 —
   because the per-row magnitude rescaling acts on the full (full-rank) base weight,
   spreading `ΔW` across hundreds of directions even though DoRA's *trainable*
   directional component is still rank ≤ r. So at this scale and task LoRA and DoRA
   are indistinguishable in update magnitude and in their effect on representations,
   but DoRA writes that change into a much higher-rank weight perturbation — a
   difference visible only once the magnitude component is measured.


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
