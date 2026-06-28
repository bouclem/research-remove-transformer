# No Residual Connections

## What we did
Removed the residual connections (`x + ...`) in the transformer block. Attention and FFN outputs replace the stream entirely instead of adding to it.

## Architecture change
```python
# Before (baseline):
x = x + self.attn(self.norm1(x))
x = x + self.ffn(self.norm2(x))

# After (no residual):
x = self.attn(self.norm1(x))
x = self.ffn(self.norm2(x))
```

## Results

| Metric | Baseline | No Residual |
|---|---|---|
| Parameters | 799,968 | 799,968 (same) |
| Best loss | 5.3731 | 5.3814 |
| Best PPL | 215.52 | 217.33 |
| Δ from baseline | — | +0.8% |
| Steps/s | 52.5 | 54.2 |

## Observations
- **Almost no degradation** — only +0.8% PPL
- Training is slightly faster (54.2 vs 52.5 steps/s) — no addition operation
- Loss is noticeably noisier — bigger swings between logged steps
- The model still converges well at 6 layers

## Interpretation
At 6 layers, the signal can flow through the sequential attention → FFN path without shortcuts. Residual connections are known to matter more as depth increases — they prevent vanishing gradients in very deep networks. At 6 layers, the network is shallow enough that gradients still flow.

The noisier loss suggests residuals do help with training stability (smoothing the optimization landscape), even if they don't affect final quality much at this depth.

This finding likely **does not scale** — at 24+ layers, removing residuals would likely cause total failure (as seen in ResNet literature). But for small transformers, they're a minor contributor.

## Plot
`plots/no_residual.png`
