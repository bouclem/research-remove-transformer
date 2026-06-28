# Single-Head Attention

## What we did
Changed `n_heads` from 6 to 1. Same d_model=96, so head_dim goes from 16 to 96. Same parameter count.

## Architecture change
```python
# Before (baseline):
n_heads = 6  # head_dim = 96/6 = 16

# After (single-head):
n_heads = 1  # head_dim = 96/1 = 96
```

## Results

| Metric | Baseline (6 heads) | Single Head |
|---|---|---|
| Parameters | 799,968 | 799,968 (same) |
| Best loss | 5.3731 | 5.4099 |
| Best PPL | 215.52 | 223.61 |
| Δ from baseline | — | +3.8% |
| Steps/s | 52.5 | 59.0 |
| head_dim | 16 | 96 |

## Observations
- **Multi-head helps by +3.8%** — not huge but consistent
- Single-head is faster (59.0 vs 52.5 steps/s) — one attention computation vs six
- Same parameter count — the difference is purely architectural
- Loss converges to a clearly higher floor

## Interpretation
Multi-head attention allows the model to attend to different aspects of the sequence simultaneously. With 6 heads, one might focus on local patterns, another on distant tokens, another on specific byte values. Single-head with head_dim=96 has richer per-head representations but can only produce one attention distribution per position.

At d_model=96, head_dim=16 is quite small — yet 6 small heads beat 1 large head. This suggests attention diversity matters more than per-head dimensionality. The ability to compute multiple independent attention patterns is valuable even at small scale.

The speed gain (+12%) doesn't compensate for the quality loss at this scale, but for latency-sensitive applications it might be an acceptable trade-off.

## Plot
`plots/single_head.png`
