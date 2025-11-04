# sbgm/evaluate/evaluate_prcp/eval_features/plot_features.py
from __future__ import annotations
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from sbgm.evaluate.evaluate_prcp.plot_utils import _ensure_dir, _nice, _savefig
from sbgm.variable_utils import get_color_for_model

SET_DPI = 300

def plot_features_all(figdir: Path, group: str, sal_metrics: dict) -> None:
    """
    Master plotting entry point for SAL decomposition.
    """
    _nice()
    fig, ax = plt.subplots(1, 1, figsize=(6, 4))
    bars = ["A", "S", "L"]
    x = np.arange(len(bars))
    width = 0.35

    col_lr = get_color_for_model("LR")
    col_gen = get_color_for_model("GEN")

    gen = [sal_metrics.get("GEN_vs_HR", {}).get(b, np.nan) for b in bars]
    lr = [sal_metrics.get("LR_vs_HR", {}).get(b, np.nan) for b in bars]

    ax.bar(x - width / 2, gen, width, label="GEN", color=col_gen)
    ax.bar(x + width / 2, lr, width, label="LR", color=col_lr)

    ax.axhline(0, color="black", lw=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(bars)
    ax.set_ylabel("Normalized difference")
    ax.set_title(f"SAL decomposition ({group})")
    ax.legend(frameon=True)
    ax.grid(True, ls=":", alpha=0.6)

    fig.tight_layout()
    _ensure_dir(figdir)
    _savefig(fig, figdir / f"features_sal_{group}.png", dpi=SET_DPI)
    plt.close(fig)