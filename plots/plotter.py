"""
Training curve plotting with matplotlib.

Generates and saves loss/perplexity plots after training.
"""

import os
import math
import matplotlib
matplotlib.use("Agg")  # non-interactive backend
import matplotlib.pyplot as plt


def plot_training_curves(
    step_losses: list[float],
    eval_losses: list[float] = None,
    eval_steps: list[int] = None,
    save_dir: str = "plots",
    title: str = "Training Curves",
) -> str:
    """
    Plot training loss (and optional eval loss) curves.

    Args:
        step_losses:  list of training loss values (logged every N steps)
        eval_losses:  list of eval loss values (optional)
        eval_steps:   list of step numbers where eval was done (optional)
        save_dir:     directory to save the plot
        title:        plot title
    Returns:
        Path to saved plot file
    """
    os.makedirs(save_dir, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # ── Loss plot ──
    ax = axes[0]
    steps = list(range(1, len(step_losses) + 1))
    ax.plot(steps, step_losses, color="#2563eb", linewidth=1.5, label="Train Loss")
    if eval_losses and eval_steps:
        ax.plot(eval_steps, eval_losses, color="#dc2626", linewidth=1.5, marker="o",
                markersize=3, label="Eval Loss")
    ax.set_xlabel("Logged Step")
    ax.set_ylabel("Loss")
    ax.set_title("Loss")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # ── Perplexity plot ──
    ax = axes[1]
    train_ppl = [math.exp(min(l, 20)) for l in step_losses]  # clamp to avoid overflow
    ax.plot(steps, train_ppl, color="#059669", linewidth=1.5, label="Train PPL")
    if eval_losses and eval_steps:
        eval_ppl = [math.exp(min(l, 20)) for l in eval_losses]
        ax.plot(eval_steps, eval_ppl, color="#dc2626", linewidth=1.5, marker="o",
                markersize=3, label="Eval PPL")
    ax.set_xlabel("Logged Step")
    ax.set_ylabel("Perplexity")
    ax.set_title("Perplexity")
    ax.legend()
    ax.grid(True, alpha=0.3)

    fig.suptitle(title, fontsize=14, fontweight="bold")
    plt.tight_layout()

    save_path = os.path.join(save_dir, "training_curves.png")
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    print(f"Training curves saved to: {save_path}")
    return save_path
