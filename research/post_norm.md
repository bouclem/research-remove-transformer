# Post-Norm vs Pre-Norm

## What we did
Swapped from pre-norm (norm before sublayer) to post-norm (norm after residual addition).

## Architecture change
```python
# Before (baseline — pre-norm):
x = x + self.attn(self.norm1(x))
x = x + self.ffn(self.norm2(x))

# After (post-norm):
x = self.norm1(x + self.attn(x))
x = self.norm2(x + self.ffn(x))
```

## Results

| Metric | Baseline (pre-norm) | Post-norm |
|---|---|---|
| Parameters | 799,968 | 799,968 (same) |
| Best PPL | 215.52 | 254.47 |
| Δ from baseline | — | +18.1% (total failure) |
| Steps/s | 52.5 | 52.8 |

## Observations
- **Post-norm completely fails** — PPL 254.47 ≈ random guessing (ln(256) = 255)
- As bad as removing normalization entirely (+18.3%)
- Loss never decreases below random level throughout training
- Same parameter count — purely architectural difference

## Interpretation
Post-norm places the normalization AFTER the residual addition. This has two critical problems:

1. **Gradient bottleneck**: In pre-norm, the residual path (x) bypasses the norm entirely — gradients flow directly to earlier layers. In post-norm, the norm sits ON the main path, so all gradients must pass through it. This constrains gradient flow.

2. **Scale mismatch**: The residual adds raw x (potentially large magnitude) to the sublayer output. The norm then must compress this combined signal. In pre-norm, the sublayer receives normalized input, producing controlled outputs that add cleanly to the residual.

With RMSNorm specifically (no mean-centering), post-norm is even worse — RMSNorm can't shift the mean, so the residual + sublayer output has an uncontrolled mean that accumulates across layers.

This explains why all modern transformers use pre-norm. The original Transformer (Vaswani et al. 2017) used post-norm with LayerNorm, but it required careful warmup and learning rate tuning. Pre-norm (introduced by Xiong et al. 2020) made training stable without these tricks.

## Plot
`plots/post_norm.png`
