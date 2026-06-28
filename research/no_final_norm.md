# No Final Norm Only

## What we did
Removed only the final RMSNorm (`norm_f`), keeping all block-level norms (norm1, norm2) intact.

## Architecture change
```python
# Before (baseline):
x = self.norm_f(x)
logits = x @ self.token_embedding.weight.T

# After (no final norm):
# x = self.norm_f(x)
logits = x @ self.token_embedding.weight.T  # raw block output
```

## Results

| Metric | Baseline | No Final Norm | No Norm (all) |
|---|---|---|---|
| Parameters | 799,968 | 798,720 (-1,248) | 798,720 (-1,248) |
| Best PPL | 215.52 | 251.15 | 254.93 |
| Δ from baseline | — | +16.5% | +18.3% |
| Steps/s | 52.5 | 54.5 | 66.8 |

## Observations
- **Final norm is nearly as critical as all norms** — +16.5% vs +18.3% for removing everything
- Only 1,248 params (0.16% of model) — same tiny cost as all norms combined
- PPL 251 is close to random guessing (255 = ln(256))
- Block norms keep intermediate layers stable but the output is still broken
- Training is slightly faster (54.5 vs 52.5 steps/s)

## Interpretation
The final norm serves a specific purpose: it normalizes the last block's output before the weight-tied output projection. Without it, the raw activations have uncontrolled scale — the dot product with embedding weights produces logits with extreme magnitudes, causing softmax saturation and near-random predictions.

The block norms (norm1, norm2) stabilize intermediate computations but don't help the final output. The model learns reasonable intermediate representations, but the "last mile" — projecting to vocab logits — requires normalized inputs.

This explains why all modern transformers include a final norm. It's not redundant with block norms; it serves a distinct purpose at the output boundary.

## Plot
`plots/no_final_norm.png`
