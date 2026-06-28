# Combo: LayerNorm + No Scaling

## What we did
Combined LayerNorm (replacing RMSNorm) with no attention scaling (removing 1/sqrt(d)).

## Results

| Metric | Baseline | LayerNorm only | No Scaling only | LayerNorm + No Scaling |
|---|---|---|---|---|
| PPL | 215.52 | 213.09 | 212.36 | 220.07 |
| Δ from baseline | — | -1.1% | -1.5% | +2.1% (worse) |
| Expected additive | — | — | — | -2.6% |
| Steps/s | 52.5 | 60.6 | 56.1 | 63.0 |

## Observations
- **Non-additive again** — individually both help, combined they hurt
- Expected -2.6% improvement, got +2.1% degradation (4.7% swing)
- This is the third non-additive combo (after no-FFN+no-RoPE and ReLU+no-scaling)
- Fastest config so far (63.0 steps/s) — LayerNorm's fused kernel + no SDPA overhead

## Interpretation
LayerNorm's mean-centering changes the activation distribution entering attention. With RMSNorm, activations are zero-mean only in aggregate (RMS normalization). LayerNorm forces each dimension to zero mean, which changes the QK dot product dynamics. Combined with no scaling, the sharper attention interacts poorly with the mean-centered inputs.

This reinforces the pattern: improvements that individually increase "sharpness" or change activation statistics don't stack. The transformer is a tightly coupled system where local improvements can create global regressions.

## Plot
`plots/layernorm_no_scaling.png`
