"""Matplotlib figures for the LoRA interpretability analyses."""

from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def plot_update_heatmap(rows: list[dict], path: str, title: str = ""):
    """Heatmap of relative update magnitude: layer (rows) x module type (cols)."""
    layers = sorted({r["layer"] for r in rows if r["layer"] >= 0})
    types = sorted({r["type"] for r in rows})
    M = np.full((len(layers), len(types)), np.nan)
    li = {l: i for i, l in enumerate(layers)}
    ti = {t: j for j, t in enumerate(types)}
    for r in rows:
        if r["layer"] in li:
            M[li[r["layer"]], ti[r["type"]]] = r["rel_update"]
    fig, ax = plt.subplots(figsize=(1.2 * len(types) + 2, 0.3 * len(layers) + 2))
    im = ax.imshow(M, aspect="auto", cmap="viridis")
    ax.set_xticks(range(len(types))); ax.set_xticklabels(types, rotation=45, ha="right")
    ax.set_yticks(range(len(layers))); ax.set_yticklabels(layers)
    ax.set_xlabel("projection"); ax.set_ylabel("layer")
    ax.set_title(title or "Relative LoRA update  ||dW|| / ||W||")
    fig.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)


def plot_update_by_layer(rows: list[dict], path: str, title: str = ""):
    """Mean relative update vs layer depth (one line per projection type)."""
    types = sorted({r["type"] for r in rows})
    fig, ax = plt.subplots(figsize=(7, 4))
    for t in types:
        pts = sorted((r["layer"], r["rel_update"]) for r in rows
                     if r["type"] == t and r["layer"] >= 0)
        if pts:
            xs, ys = zip(*pts)
            ax.plot(xs, ys, marker="o", ms=3, label=t)
    ax.set_xlabel("layer"); ax.set_ylabel("||dW|| / ||W||")
    ax.set_title(title or "LoRA update magnitude across depth")
    ax.legend(fontsize=8); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)


def plot_drift(drift: dict, path: str, title: str = ""):
    """Per-layer representation drift (cosine similarity and relative L2)."""
    layers = drift["layer"]
    fig, ax1 = plt.subplots(figsize=(7, 4))
    ax1.plot(layers, drift["cosine_similarity"], color="#1f77b4", marker="o", ms=3)
    ax1.set_xlabel("layer (0 = embeddings)")
    ax1.set_ylabel("cosine(adapter on, off)", color="#1f77b4")
    ax2 = ax1.twinx()
    ax2.plot(layers, drift["relative_l2"], color="#d62728", marker="s", ms=3)
    ax2.set_ylabel("relative L2 drift", color="#d62728")
    ax1.set_title(title or "Representation drift induced by LoRA")
    ax1.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)
