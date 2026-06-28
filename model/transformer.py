"""
800K parameter decoder-only Transformer.

Architecture:
  - vocab_size:  256  (byte-level tokenizer)
  - d_model:      96
  - n_layers:      6
  - n_heads:       6  (head_dim = 16)
  - d_ff:        480
  - max_seq_len: 256
  - RoPE positional encoding (0 params)
  - RMSNorm pre-normalization (no bias)
  - GELU activation in FFN
  - Weight tying: token embedding == output projection
  - No biases on linear layers

Total parameters: 799,968 (~800K)
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ── RMSNorm ──────────────────────────────────────────────────────────────────

class RMSNorm(nn.Module):
    """Root Mean Square LayerNorm — no bias, only learnable scale."""

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        norm = x.pow(2).mean(dim=-1, keepdim=True)
        x_normed = x * torch.rsqrt(norm + self.eps)
        return self.weight * x_normed


# ── Rotary Position Embedding ────────────────────────────────────────────────

def precompute_rope_cache(head_dim: int, max_seq_len: int, base: float = 10000.0):
    """Precompute sin/cos tables for RoPE."""
    inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2).float() / head_dim))
    positions = torch.arange(max_seq_len).float()
    freqs = torch.outer(positions, inv_freq)          # (seq, head_dim/2)
    cos = freqs.cos()                                   # (seq, head_dim/2)
    sin = freqs.sin()                                   # (seq, head_dim/2)
    # Duplicate to match head_dim: (seq, head_dim)
    cos = torch.cat([cos, cos], dim=-1)
    sin = torch.cat([sin, sin], dim=-1)
    return cos, sin


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """
    Apply rotary embeddings to tensor x.
    x:   (batch, n_heads, seq, head_dim)
    cos: (seq, head_dim)
    sin: (seq, head_dim)
    """
    seq_len = x.shape[2]
    cos = cos[:seq_len].unsqueeze(0).unsqueeze(0)  # (1, 1, seq, head_dim)
    sin = sin[:seq_len].unsqueeze(0).unsqueeze(0)

    # Rotate half
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]

    # Interleave approach: rotate pairs (x_even, x_odd)
    # We use the "rotate half" variant: concat(-x2, x1) then apply cos/sin
    x_rot = torch.cat([-x2, x1], dim=-1)

    return x * cos + x_rot * sin


# ── Multi-Head Self-Attention ────────────────────────────────────────────────

class CausalSelfAttention(nn.Module):
    """Multi-head causal self-attention with RoPE, no bias."""

    def __init__(self, d_model: int, n_heads: int, max_seq_len: int):
        super().__init__()
        assert d_model % n_heads == 0
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.d_model = d_model

        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.proj = nn.Linear(d_model, d_model, bias=False)

        # Precompute RoPE cache (not a parameter)
        cos, sin = precompute_rope_cache(self.head_dim, max_seq_len)
        self.register_buffer("rope_cos", cos, persistent=False)
        self.register_buffer("rope_sin", sin, persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape

        # Project to Q, K, V and reshape to (B, n_heads, T, head_dim)
        qkv = self.qkv(x)
        q, k, v = qkv.chunk(3, dim=-1)
        q = q.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)

        # Apply RoPE to Q and K
        q = apply_rope(q, self.rope_cos, self.rope_sin)
        k = apply_rope(k, self.rope_cos, self.rope_sin)

        # Scaled dot-product attention with causal mask
        # ABLATION: LayerNorm + no scaling combo — uncomment SDPA to restore
        # Manual attention without 1/sqrt(d) scaling:
        scores = q @ k.transpose(-2, -1)  # (B, n_heads, T, T) — no scaling
        causal_mask = torch.tril(torch.ones(T, T, device=x.device, dtype=torch.bool))
        scores = scores.masked_fill(~causal_mask, float('-inf'))
        attn = F.softmax(scores, dim=-1) @ v
        # attn = F.scaled_dot_product_attention(q, k, v, is_causal=True)

        # Reshape back to (B, T, C)
        attn = attn.transpose(1, 2).contiguous().view(B, T, C)
        return self.proj(attn)


# ── Feed-Forward Network ─────────────────────────────────────────────────────

class FeedForward(nn.Module):
    """Standard FFN: up -> GELU -> down, no bias."""

    def __init__(self, d_model: int, d_ff: int):
        super().__init__()
        self.up = nn.Linear(d_model, d_ff, bias=False)
        self.down = nn.Linear(d_ff, d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down(F.gelu(self.up(x)))


# ── Transformer Block ────────────────────────────────────────────────────────

class TransformerBlock(nn.Module):
    """Pre-norm transformer block: RMSNorm -> Attn -> residual, RMSNorm -> FFN -> residual."""

    def __init__(self, d_model: int, n_heads: int, d_ff: int, max_seq_len: int):
        super().__init__()
        # ABLATION: LayerNorm + no scaling combo — change back to RMSNorm and SDPA to restore
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = CausalSelfAttention(d_model, n_heads, max_seq_len)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = FeedForward(d_model, d_ff)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))
        return x


# ── Full Model ───────────────────────────────────────────────────────────────

class TransformerConfig:
    """Configuration for the 800K parameter transformer."""

    def __init__(
        self,
        vocab_size: int = 256,
        d_model: int = 96,
        n_layers: int = 6,
        n_heads: int = 6,
        d_ff: int = 480,
        max_seq_len: int = 256,
    ):
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.d_ff = d_ff
        self.max_seq_len = max_seq_len


class TransformerLM(nn.Module):
    """Decoder-only transformer language model with ~800K parameters."""

    def __init__(self, config: TransformerConfig = None):
        super().__init__()
        if config is None:
            config = TransformerConfig()
        self.config = config

        # Token embedding (shared with output projection)
        self.token_embedding = nn.Embedding(config.vocab_size, config.d_model)

        # Transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(
                d_model=config.d_model,
                n_heads=config.n_heads,
                d_ff=config.d_ff,
                max_seq_len=config.max_seq_len,
            )
            for _ in range(config.n_layers)
        ])

        # Final norm
        # ABLATION: LayerNorm + no scaling combo — change back to RMSNorm to restore
        self.norm_f = nn.LayerNorm(config.d_model)

        # Weight tying: output projection shares embedding weights
        # No separate lm_head — we use token_embedding.weight for projection

        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, (RMSNorm, nn.LayerNorm)):
            nn.init.ones_(module.weight)

    def forward(self, idx: torch.Tensor, targets: torch.Tensor = None):
        """
        Args:
            idx:     (batch, seq) token indices
            targets: (batch, seq) target token indices (optional)

        Returns:
            logits: (batch, seq, vocab_size)
            loss:   scalar (if targets provided)
        """
        B, T = idx.shape
        assert T <= self.config.max_seq_len, f"Sequence length {T} exceeds max {self.config.max_seq_len}"

        # Token embedding (no positional embedding — RoPE handles positions)
        x = self.token_embedding(idx)  # (B, T, d_model)

        # Transformer blocks
        for block in self.blocks:
            x = block(x)

        # Final norm
        x = self.norm_f(x)

        # Output projection (weight-tied with embedding)
        logits = x @ self.token_embedding.weight.T  # (B, T, vocab_size)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.view(-1, self.config.vocab_size),
                targets.view(-1),
            )

        return logits, loss

    @torch.no_grad()
    def generate(
        self,
        idx: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 1.0,
        top_k: int | None = None,
    ) -> torch.Tensor:
        """
        Autoregressive generation with optional temperature and top-k sampling.

        Args:
            idx:            (1, seq) starting token indices
            max_new_tokens: number of tokens to generate
            temperature:    sampling temperature (1.0 = no change)
            top_k:          if set, restrict to top-k logits
        Returns:
            (1, seq + max_new_tokens) token indices
        """
        self.eval()
        for _ in range(max_new_tokens):
            # Crop to max_seq_len
            idx_cond = idx if idx.shape[1] <= self.config.max_seq_len else idx[:, -self.config.max_seq_len:]

            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / temperature

            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.shape[-1]))
                logits[logits < v[:, [-1]]] = float("-inf")

            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            idx = torch.cat([idx, next_token], dim=1)

        return idx

    def count_parameters(self) -> dict:
        """Return detailed parameter count breakdown."""
        details = {}

        # Token embedding
        details["token_embedding"] = self.token_embedding.weight.numel()

        # Per-block
        for i, block in enumerate(self.blocks):
            details[f"block_{i}/qkv"] = block.attn.qkv.weight.numel()
            details[f"block_{i}/proj"] = block.attn.proj.weight.numel()
            details[f"block_{i}/norm1"] = block.norm1.weight.numel()
            details[f"block_{i}/ffn_up"] = block.ffn.up.weight.numel()
            details[f"block_{i}/ffn_down"] = block.ffn.down.weight.numel()
            details[f"block_{i}/norm2"] = block.norm2.weight.numel()

        # Final norm
        details["norm_f"] = self.norm_f.weight.numel()

        total = sum(p.numel() for p in self.parameters())
        details["total"] = total
        details["total_formatted"] = f"{total:,}"
        return details


# ── Convenience ──────────────────────────────────────────────────────────────

def create_model() -> TransformerLM:
    """Create the 800K parameter transformer."""
    config = TransformerConfig()
    return TransformerLM(config)


if __name__ == "__main__":
    model = create_model()
    details = model.count_parameters()

    print("=" * 60)
    print("800K Parameter Transformer — Parameter Breakdown")
    print("=" * 60)

    # Group by component
    emb = details["token_embedding"]
    attn_total = sum(v for k, v in details.items() if "qkv" in k or "proj" in k)
    ffn_total = sum(v for k, v in details.items() if "ffn" in k)
    norm_total = sum(v for k, v in details.items() if "norm" in k)

    print(f"\nToken embedding:     {emb:>10,}")
    print(f"Attention (all):     {attn_total:>10,}")
    print(f"FFN (all):           {ffn_total:>10,}")
    print(f"RMSNorm (all):       {norm_total:>10,}")
    print(f"\n{'TOTAL':>20} {details['total']:>10,}")
    print(f"\nConfig: d_model={model.config.d_model}, n_layers={model.config.n_layers}, "
          f"n_heads={model.config.n_heads}, d_ff={model.config.d_ff}, "
          f"vocab={model.config.vocab_size}")

    # Quick forward pass test
    print("\n--- Forward pass test ---")
    x = torch.randint(0, 256, (2, 32))
    logits, loss = model(x, targets=x)
    print(f"Input shape:  {x.shape}")
    print(f"Logits shape: {logits.shape}")
    print(f"Loss:         {loss.item():.4f}")

    # Generation test
    print("\n--- Generation test ---")
    prompt = torch.randint(0, 256, (1, 8))
    out = model.generate(prompt, max_new_tokens=16, temperature=1.0, top_k=50)
    print(f"Prompt shape: {prompt.shape}")
    print(f"Output shape: {out.shape}")
    print(f"Generated tokens: {out[0].tolist()}")
