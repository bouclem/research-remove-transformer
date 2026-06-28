# A9: Attention Scaling Removed

## What we did
Removed the `1/sqrt(head_dim)` scaling from the QK dot product. Replaced PyTorch's `scaled_dot_product_attention` with manual attention that skips the scaling step.

## Architecture change
```python
# Before (baseline):
attn = F.scaled_dot_product_attention(q, k, v, is_causal=True)  # scales by 1/sqrt(d)

# After (no scaling):
scores = q @ k.transpose(-2, -1)  # raw dot product, no 1/sqrt(d)
scores = scores.masked_fill(~causal_mask, float('-inf'))
attn = F.softmax(scores, dim=-1) @ v
```

## Results

| Metric | Baseline (scaled) | No Scaling |
|---|---|---|
| Parameters | 799,968 | 799,968 (same) |
| Best loss | 5.3731 | 5.3583 |
| Best PPL | 215.52 | 212.36 |
| Δ from baseline | — | -1.5% (better) |
| Steps/s | 52.5 | 56.1 |

## Observations
- **Removing scaling improves PPL** by 1.5% — second improvement after ReLU
- Training is faster (56.1 vs 52.5 steps/s) — manual attention avoids SDPA overhead at small sizes
- No instability — softmax still works fine with larger logits
- Loss converges lower than baseline throughout training

## Interpretation
The `1/sqrt(head_dim)` scaling was introduced to keep QK dot product variance ~1 at large head_dim, preventing softmax saturation. With head_dim=16, the scaling factor is `1/4 = 0.25` — quite aggressive. It dampens the logits too much, leading to overly uniform attention distributions.

Without scaling, the sharper softmax creates more decisive attention patterns. At head_dim=16, the raw dot products don't explode enough to cause saturation, so the scaling is unnecessary — and harmful.

This finding is likely head_dim-dependent. At head_dim=64 or 128 (common in larger models), removing scaling would likely cause softmax saturation and hurt performance. But at head_dim=16, the model benefits from sharper attention.

## Plot
`plots/no_scaling.png`
