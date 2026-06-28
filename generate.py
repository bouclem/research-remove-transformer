"""
Text generation script for the 800K parameter transformer.

Loads a trained checkpoint and generates text autoregressively.

Usage:
    python generate.py --checkpoint checkpoints/best.pt --prompt "Hello world"
    python generate.py --checkpoint checkpoints/best.pt --prompt "" --max_tokens 200 --temperature 0.8
"""

import argparse
import torch

from model import TransformerLM, TransformerConfig
from tokenizer import ByteTokenizer


def load_model(checkpoint_path: str, device: str = "auto") -> tuple[TransformerLM, ByteTokenizer]:
    """Load model from checkpoint."""
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device)

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    config = TransformerConfig(**checkpoint["config"])
    model = TransformerLM(config).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    print(f"Loaded checkpoint: {checkpoint_path}")
    print(f"  Step: {checkpoint.get('step', '?')}")
    print(f"  Loss:  {checkpoint.get('loss', '?'):.4f}")
    print(f"  Dataset: {checkpoint.get('dataset', '?')}")
    print(f"  Params: {sum(p.numel() for p in model.parameters()):,}")

    return model, ByteTokenizer()


def generate_text(
    model: TransformerLM,
    tokenizer: ByteTokenizer,
    prompt: str,
    max_tokens: int = 200,
    temperature: float = 1.0,
    top_k: int = 50,
    device: torch.device = None,
) -> str:
    """Generate text from a prompt."""
    if device is None:
        device = next(model.parameters()).device

    # Encode prompt
    prompt_ids = tokenizer.encode(prompt)
    if len(prompt_ids) == 0:
        prompt_ids = [tokenizer.bos_id]

    idx = torch.tensor([prompt_ids], dtype=torch.long, device=device)

    print(f"\nPrompt: {prompt!r} ({len(prompt_ids)} tokens)")
    print(f"Generating {max_tokens} tokens (temp={temperature}, top_k={top_k})...")
    print("-" * 60)

    with torch.no_grad():
        out_ids = model.generate(
            idx,
            max_new_tokens=max_tokens,
            temperature=temperature,
            top_k=top_k,
        )

    # Decode
    generated_ids = out_ids[0].tolist()
    full_text = tokenizer.decode(generated_ids)
    new_text = tokenizer.decode(generated_ids[len(prompt_ids):])

    print(f"Full output ({len(generated_ids)} tokens):")
    print(full_text)
    print("-" * 60)
    print(f"Generated text only:")
    print(new_text)

    return full_text


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate text with trained transformer")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to .pt checkpoint")
    parser.add_argument("--prompt", type=str, default="", help="Text prompt")
    parser.add_argument("--max_tokens", type=int, default=200)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top_k", type=int, default=50)
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    args = parser.parse_args()

    model, tokenizer = load_model(args.checkpoint, args.device)
    generate_text(
        model, tokenizer,
        prompt=args.prompt,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
    )
