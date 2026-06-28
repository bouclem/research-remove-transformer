# No Block Norms (Final Norm Kept)

## What we did
Removed block-level RMSNorm (norm1, norm2) from all transformer blocks, keeping only the final norm (norm_f).

## Architecture change
```python
# Before (baseline):
x = x + self.attn(self.norm1(x))
x = x + self.ffn(self.norm2(x))

# After (no block norms):
x = x + self.attn(x)
x = x + self.ffn(x)
```

## Results

| Metric | Baseline | No Norm (all) | No Final Norm | No Block Norms |
|---|---|---|---|---|
| Parameters | 799,968 | 798,720 | 798,720 | 798,720 |
| Best PPL | 215.52 | 254.93 | 251.15 | 253.49 |
| Δ from baseline | — | +18.3% | +16.5% | +17.6% |
| Steps/s | 52.5 | 66.8 | 54.5 | 65.7 |

## Observations
- **Block norms are critical** — removing them costs +17.6% PPL
- Nearly as bad as removing all norms (+18.3%)
- The final norm alone can't save the model — intermediate activations explode through 6 layers
- Training is fast (65.7 steps/s) — fewer norm operations, but quality is terrible

## Interpretation
The full normalization picture:

| Config | PPL | Δ |
|---|---|---|
| All norms (baseline) | 215.52 | — |
| No final norm only | 251.15 | +16.5% |
| No block norms only | 253.49 | +17.6% |
| No norms at all | 254.93 | +18.3% |

Block norms and final norm serve **different but equally essential roles**:
- **Block norms**: prevent activation explosion through 6 layers of attention + FFN. Without them, each layer amplifies the previous layer's magnitude, creating an unstable cascade.
- **Final norm**: normalizes the last block's output for the weight-tied projection. Without it, logits have extreme magnitudes.

Removing either alone accounts for ~90% of the total normalization impact. They're not redundant — they're complementary. But the model needs both to function.

## Plot
`plots/no_block_norm.png`
