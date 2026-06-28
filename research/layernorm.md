# LayerNorm vs RMSNorm

## What we did
Swapped all RMSNorm instances to PyTorch's built-in `nn.LayerNorm`. LayerNorm adds a learnable bias term (mean-centering) that RMSNorm lacks.

## Architecture change
```python
# Before (baseline — RMSNorm):
self.norm1 = RMSNorm(d_model)  # only scale, no bias

# After (LayerNorm):
self.norm1 = nn.LayerNorm(d_model)  # scale + bias
```

## Results

| Metric | Baseline (RMSNorm) | LayerNorm |
|---|---|---|
| Parameters | 799,968 | 801,216 (+1,248) |
| Best loss | 5.3731 | 5.3617 |
| Best PPL | 215.52 | 213.09 |
| Δ from baseline | — | -1.1% (better) |
| Steps/s | 52.5 | 60.6 |

## Observations
- **LayerNorm is slightly better** — PPL 213.09 vs 215.52
- Only +1,248 params (the bias terms) — negligible cost
- Training is significantly faster (60.6 vs 52.5 steps/s, +15%)
- Loss curve is similar shape but consistently slightly lower

## Interpretation
RMSNorm normalizes by RMS only (no mean subtraction). LayerNorm subtracts the mean AND normalizes by std, plus has a bias term. The mean-centering gives each dimension an independent shift, allowing better representation calibration.

The speed difference is likely because `nn.LayerNorm` is a fused C++ CUDA kernel, while the custom `RMSNorm` uses multiple Python-level tensor operations. This is an implementation artifact, not a theoretical difference — a fused RMSNorm would be comparable.

The improvement is small but consistent. The extra 1,248 bias params are worth it. Combined with the speed gain from the fused kernel, LayerNorm is a practical win at this scale.

## Plot
`plots/layernorm.png`
