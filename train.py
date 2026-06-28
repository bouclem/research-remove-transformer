"""
Training script for the 800K parameter transformer.

Step-based training (no epochs). Supports dataset selection and
automatic training curve plotting with matplotlib.

Usage:
    python train.py --dataset custom --steps 5000 --batch_size 16
    python train.py --dataset wikitext2 --steps 10000 --device cuda
"""

import argparse
import os
import time
import math
import torch

from model import TransformerLM, TransformerConfig
from datasets import create_infinite_loader, DATASETS
from plots import plot_training_curves


def train(
    dataset_name: str,
    max_steps: int = 5000,
    batch_size: int = 32,
    seq_len: int = 256,
    lr: float = 3e-4,
    weight_decay: float = 0.1,
    warmup_steps: int = 100,
    grad_clip: float = 1.0,
    device: str = "auto",
    save_dir: str = "checkpoints",
    save_every: int = 500,
    log_every: int = 50,
    eval_every: int = 500,
    eval_steps: int = 50,
    plot_dir: str = "plots",
    custom_path: str = None,
    experiment_name: str = "baseline",
):
    """Full step-based training pipeline."""

    # ── Device selection ──
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device)
    print(f"Device: {device}")

    # ── Load training data ──
    print(f"\nDataset: {dataset_name}")
    train_loader = create_infinite_loader(dataset_name, seq_len, batch_size, "train", custom_path)

    # ── Eval dataset (optional) ──
    eval_loader = None
    if eval_every > 0:
        try:
            eval_loader = create_infinite_loader(dataset_name, seq_len, batch_size, "validation", custom_path)
        except Exception as e:
            print(f"  (no eval split available: {e})")
            eval_loader = None

    # ── Model ──
    config = TransformerConfig(max_seq_len=seq_len)
    model = TransformerLM(config).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {total_params:,}")

    # ── Optimizer (AdamW) ──
    decay_params = [p for n, p in model.named_parameters() if p.dim() >= 2]
    nodecay_params = [p for n, p in model.named_parameters() if p.dim() < 2]
    optim_groups = [
        {"params": decay_params, "weight_decay": weight_decay},
        {"params": nodecay_params, "weight_decay": 0.0},
    ]
    optimizer = torch.optim.AdamW(optim_groups, lr=lr, betas=(0.9, 0.95))

    # ── Learning rate scheduler (cosine with warmup) ──
    def get_lr(step: int) -> float:
        if step < warmup_steps:
            return lr * (step + 1) / warmup_steps
        decay = max(0.1, (step - warmup_steps) / max(1, max_steps - warmup_steps))
        return lr * 0.5 * (1.0 + math.cos(math.pi * decay))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, get_lr)

    # ── Mixed precision ──
    use_amp = device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda") if use_amp else None

    # ── Checkpoint dir ──
    os.makedirs(save_dir, exist_ok=True)

    # ── Training ──
    step_losses = []
    eval_losses = []
    eval_step_marks = []
    best_loss = float("inf")
    t0 = time.time()

    print(f"\nStarting training: {max_steps} steps")
    print("=" * 70)

    model.train()
    for step in range(1, max_steps + 1):
        x, y = train_loader.next()
        x, y = x.to(device), y.to(device)

        # Forward
        if use_amp:
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                logits, loss = model(x, targets=y)
        else:
            logits, loss = model(x, targets=y)

        # Backward
        optimizer.zero_grad()
        if use_amp:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()
        scheduler.step()

        step_losses.append(loss.item())

        # Logging
        if step % log_every == 0 or step == 1:
            current_lr = scheduler.get_last_lr()[0]
            elapsed = time.time() - t0
            steps_per_sec = step / elapsed
            print(
                f"  Step {step:5d}/{max_steps} | "
                f"Loss {loss.item():.4f} | "
                f"PPL {math.exp(loss.item()):.2f} | "
                f"LR {current_lr:.2e} | "
                f"{steps_per_sec:.1f} steps/s"
            )

        # Evaluation
        if eval_loader is not None and eval_every > 0 and step % eval_every == 0:
            model.eval()
            eval_loss_sum = 0.0
            for _ in range(eval_steps):
                xe, ye = eval_loader.next()
                xe, ye = xe.to(device), ye.to(device)
                with torch.no_grad():
                    _, eloss = model(xe, targets=ye)
                    eval_loss_sum += eloss.item()
            avg_eval_loss = eval_loss_sum / eval_steps
            eval_losses.append(avg_eval_loss)
            eval_step_marks.append(step)
            print(
                f"  [EVAL] Step {step:5d} | "
                f"Eval Loss {avg_eval_loss:.4f} | "
                f"Eval PPL {math.exp(avg_eval_loss):.2f}"
            )
            model.train()

        # Checkpointing
        if step % save_every == 0 or step == max_steps:
            avg_recent = sum(step_losses[-save_every:]) / min(save_every, len(step_losses))
            torch.save({
                "model_state": model.state_dict(),
                "config": config.__dict__,
                "step": step,
                "loss": avg_recent,
                "dataset": dataset_name,
            }, os.path.join(save_dir, f"step_{step}.pt"))

            if avg_recent < best_loss:
                best_loss = avg_recent
                torch.save({
                    "model_state": model.state_dict(),
                    "config": config.__dict__,
                    "step": step,
                    "loss": avg_recent,
                    "dataset": dataset_name,
                }, os.path.join(save_dir, "best.pt"))

    # ── Final summary ──
    elapsed = time.time() - t0
    print("=" * 70)
    print(f"Training complete in {elapsed:.1f}s ({max_steps/elapsed:.1f} steps/s)")
    print(f"Best loss: {best_loss:.4f} (PPL: {math.exp(best_loss):.2f})")
    print(f"Best checkpoint: {os.path.join(save_dir, 'best.pt')}")

    # ── Plot training curves ──
    plot_title = f"800K Transformer — {experiment_name} — {dataset_name} — {max_steps} steps"
    plot_training_curves(
        step_losses=step_losses,
        eval_losses=eval_losses if eval_losses else None,
        eval_steps=eval_step_marks if eval_step_marks else None,
        save_dir=plot_dir,
        title=plot_title,
        experiment_name=experiment_name,
    )


# ── CLI ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train 800K transformer (step-based)")
    parser.add_argument("--dataset", type=str, default="custom",
                        choices=list(DATASETS.keys()),
                        help="Dataset name")
    parser.add_argument("--steps", type=int, default=5000, help="Total training steps")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--seq_len", type=int, default=256)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight_decay", type=float, default=0.1)
    parser.add_argument("--warmup_steps", type=int, default=100)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--save_dir", type=str, default="checkpoints")
    parser.add_argument("--save_every", type=int, default=500)
    parser.add_argument("--log_every", type=int, default=50)
    parser.add_argument("--eval_every", type=int, default=500, help="Eval every N steps (0=disable)")
    parser.add_argument("--eval_steps", type=int, default=50)
    parser.add_argument("--plot_dir", type=str, default="plots")
    parser.add_argument("--custom_path", type=str, default=None, help="Override path for custom dataset")
    parser.add_argument("--experiment_name", type=str, default="baseline", help="Name for plot filename")
    args = parser.parse_args()

    train(
        dataset_name=args.dataset,
        max_steps=args.steps,
        batch_size=args.batch_size,
        seq_len=args.seq_len,
        lr=args.lr,
        weight_decay=args.weight_decay,
        warmup_steps=args.warmup_steps,
        grad_clip=args.grad_clip,
        device=args.device,
        save_dir=args.save_dir,
        save_every=args.save_every,
        log_every=args.log_every,
        eval_every=args.eval_every,
        eval_steps=args.eval_steps,
        plot_dir=args.plot_dir,
        custom_path=args.custom_path,
        experiment_name=args.experiment_name,
    )
