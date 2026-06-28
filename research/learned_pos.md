# Learned Positional Embeddings vs RoPE

## What we did
Removed RoPE from attention and added learned positional embeddings (`nn.Embedding(max_seq_len, d_model)`) added to token embeddings at the input.

## Architecture change
```python
# Before (baseline — RoPE):
# In CausalSelfAttention:
q = apply_rope(q, self.rope_cos, self.rope_sin)
k = apply_rope(k, self.rope_cos, self.rope_sin)

# In TransformerLM.forward:
x = self.token_embedding(idx)  # no positional info in x

# After (learned positional embeddings):
# RoPE removed from attention
# In TransformerLM.__init__:
self.pos_embedding = nn.Embedding(max_seq_len, d_model)

# In TransformerLM.forward:
x = self.token_embedding(idx)
pos = torch.arange(T, device=idx.device)
x = x + self.pos_embedding(pos)
```

## Results

| Metric | Baseline (RoPE) | No RoPE | Learned Pos |
|---|---|---|---|
| Parameters | 799,968 | 799,968 | 831,488 (+31,520) |
| Best PPL | 215.52 | 225.24 | 234.48 |
| Δ from baseline | — | +4.5% | +8.8% |
| Steps/s | 52.5 | 56.1 | 60.5 |

## Observations
- **Learned pos is WORSE than no positional encoding** — +8.8% vs +4.5%
- Adds 31,520 params (max_seq_len=256 × d_model=96) — 4% more params
- Training is faster (60.5 vs 52.5 steps/s) — no RoPE computation
- Loss converges to a much higher floor than both RoPE and no-pos

## Interpretation
This is a surprising result: learned positional embeddings are worse than having NO positional encoding. Several factors:

1. **Data scarcity**: With only 9.6KB of training data, the learned embeddings can't be trained well. Each position (0-127) is seen relatively few times, so the embeddings remain noisy.

2. **Initialization interference**: The pos embeddings are initialized with std=0.02 (same as token embeddings). When added to token embeddings, they inject noise that confuses the model early in training.

3. **No extrapolation**: RoPE generalizes to any sequence length via its mathematical rotation. Learned embeddings only work for positions seen during training.

4. **Parameter inefficiency**: 31K params spent on positional info that RoPE provides for free. These params compete with the rest of the model for gradient signal.

RoPE's elegance: it encodes positions as rotations in the attention space, requiring zero parameters and generalizing to any length. At small scale with limited data, this mathematical prior beats learned representations.

## Plot
`plots/learned_pos.png`
