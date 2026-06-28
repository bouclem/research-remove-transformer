# A2: RoPE (Positional Encoding) Removed

## What we did
Commented out the Rotary Position Embedding (RoPE) application in `CausalSelfAttention`. Q and K are used raw — no rotational encoding of position.

## Architecture change
```python
# Before (baseline):
q = apply_rope(q, self.rope_cos, self.rope_sin)
k = apply_rope(k, self.rope_cos, self.rope_sin)

# After (no RoPE):
# q = apply_rope(q, self.rope_cos, self.rope_sin)  # commented out
# k = apply_rope(k, self.rope_cos, self.rope_sin)  # commented out
```

## Results

| Metric | Baseline | No RoPE |
|---|---|---|
| Parameters | 799,968 | 799,968 (same — RoPE has 0 params) |
| Best loss | 5.3731 | 5.4172 |
| Best PPL | 215.52 | 225.24 |
| Steps/s | 52.5 | 61.0 |
| Training time | 38.1s | 32.8s |

## Observations
- PPL only degrades by ~4.5% — surprisingly mild
- Training is faster (61.0 vs 52.5 steps/s) since RoPE rotation is skipped
- The model still converges and learns structure
- Loss curve is close to baseline, slight gap throughout

## Interpretation
The causal mask itself provides implicit positional information: a token at position *t* can only attend to positions ≤ *t*, so the attention pattern is inherently position-aware. At seq_len=128, this implicit signal may be enough. RoPE likely matters more at longer sequence lengths where distinguishing distant positions becomes critical.

The question "do transformers need positional encoding?" — at this scale, the answer is **no, but it helps**.

## Plot
`plots/no_rope.png`
