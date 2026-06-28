# A1: Normalization Removed

## What we did
Commented out all RMSNorm instances — `norm1` and `norm2` in each transformer block, plus `norm_f` (final norm). Attention and FFN operate on raw residual stream values.

## Architecture change
```python
# Before (baseline):
x = x + self.attn(self.norm1(x))
x = x + self.ffn(self.norm2(x))
# ... later: x = self.norm_f(x)

# After (no norm):
x = x + self.attn(x)
x = x + self.ffn(x)
# ... later: (no final norm)
```

## Results

| Metric | Baseline | No Norm |
|---|---|---|
| Parameters | 799,968 | 798,720 (-1,248) |
| Best loss | 5.3731 | 5.5410 |
| Best PPL | 215.52 | 254.93 |
| Steps/s | 52.5 | 66.9 |
| Training time | 38.1s | 29.9s |

## Observations
- **The model does not learn at all.** Loss is flat at ~5.54 from step 50 to 2000
- PPL 255 ≈ random guessing over vocab=256 (ln(256) = 5.545)
- Training is faster (66.9 vs 52.5 steps/s) but completely pointless
- The 1,248 parameter savings is negligible — normalization is almost free

## Interpretation
Without normalization, residual connections cause activation magnitudes to grow unboundedly across 6 layers. The attention QK products become dominated by magnitude rather than content, and FFN activations saturate. Gradients either vanish or explode, preventing any learning.

This is the strongest finding so far: **normalization is not optional**. Unlike FFN (-69% params, -9% PPL) or RoPE (0 params, -4.5% PPL), removing normalization destroys the model entirely for a negligible parameter saving.

## Plot
`plots/no_norm.png`
