"""
Compare all experiment plots side by side.

Scans plots/ for *.png files and creates a comparison grid.
Also generates an overlay plot showing all loss curves on the same axes.

Usage:
    python compare.py
"""

import os
import math
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.image as mpimg


PLOTS_DIR = "plots"
OUTPUT_DIR = "plots"


def compare_grid():
    """Create a grid of all experiment plots side by side."""
    pngs = sorted([f for f in os.listdir(PLOTS_DIR) if f.endswith(".png") and f != "comparison.png" and f != "overlay.png"])
    if not pngs:
        print("No plots found in plots/")
        return

    n = len(pngs)
    cols = min(3, n)
    rows = math.ceil(n / cols)

    fig, axes = plt.subplots(rows, cols, figsize=(7 * cols, 5 * rows))
    if n == 1:
        axes = [axes]
    else:
        axes = axes.flatten() if hasattr(axes, "flatten") else [axes]

    for i, png in enumerate(pngs):
        img = mpimg.imread(os.path.join(PLOTS_DIR, png))
        axes[i].imshow(img)
        axes[i].set_title(png.replace(".png", ""), fontsize=12, fontweight="bold")
        axes[i].axis("off")

    # Hide unused subplots
    for i in range(n, len(axes)):
        axes[i].axis("off")

    fig.suptitle("All Ablation Experiments — Training Curves", fontsize=16, fontweight="bold")
    plt.tight_layout()

    save_path = os.path.join(OUTPUT_DIR, "comparison.png")
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Comparison grid saved to: {save_path}")


def compare_table():
    """Print a summary table of all experiments."""
    experiments = {
        "baseline":       {"params": 799_968,  "ppl": 215.52, "note": "Full model"},
        "no_ffn":         {"params": 246_432,  "ppl": 234.84, "note": "FFN removed"},
        "no_rope":        {"params": 799_968,  "ppl": 225.24, "note": "RoPE removed"},
        "no_norm":        {"params": 798_720,  "ppl": 254.93, "note": "Normalization removed"},
        "no_ffn_no_rope": {"params": 246_432,  "ppl": 235.43, "note": "FFN + RoPE removed"},
        "no_residual":    {"params": 799_968,  "ppl": 217.33, "note": "Residual connections removed"},
        "relu":           {"params": 799_968,  "ppl": 213.25, "note": "GELU -> ReLU"},
        "swiglu":         {"params": 1_076_448, "ppl": 222.11, "note": "GELU -> SwiGLU"},
    }

    baseline_ppl = experiments["baseline"]["ppl"]

    print("\n" + "=" * 85)
    print(f"{'Experiment':<20} {'Params':>12} {'PPL':>10} {'Δ PPL':>10} {'Note'}")
    print("=" * 85)

    for name, data in sorted(experiments.items(), key=lambda x: x[1]["ppl"]):
        delta = ((data["ppl"] - baseline_ppl) / baseline_ppl) * 100
        delta_str = f"{delta:+.1f}%" if name != "baseline" else "—"
        print(f"{name:<20} {data['params']:>12,} {data['ppl']:>10.2f} {delta_str:>10}   {data['note']}")

    print("=" * 85)
    print(f"{'Best: relu':<20} {experiments['relu']['params']:>12,} {experiments['relu']['ppl']:>10.2f} {'-1.1%':>10}   GELU -> ReLU")
    print()


if __name__ == "__main__":
    compare_table()
    compare_grid()
