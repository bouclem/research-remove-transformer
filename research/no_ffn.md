# A-FFN: Feed-Forward Network Removed

## What we did
Commented out the FFN (FeedForward module) in `TransformerBlock`. The model is attention-only — no `up -> GELU -> down` projection.

## Architecture change
```python
# Before (baseline):
x = x + self.attn(self.norm1(x))
x = x + self.ffn(self.norm2(x))

# After (no FFN):
x = x + self.attn(self.norm1(x))
# x = x + self.ffn(self.norm2(x))  # commented out
```

## Results

| Metric | Baseline | No FFN |
|---|---|---|
| Parameters | 799,968 | 246,432 (-69%) |
| Best loss | 5.3731 | 5.4589 |
| Best PPL | 215.52 | 234.84 |
| Steps/s | 52.5 | 67.6 |
| Training time | 38.1s | 29.6s |

## Observations
- FFN accounts for 69% of all parameters (552,960 of 799,968) but removing it only costs ~9% PPL
- Training is faster (67.6 vs 52.5 steps/s) due to fewer matmuls
- The model still learns — attention alone captures meaningful structure
- Loss curve is flatter, suggesting less capacity to memorize patterns

## Interpretation
The FFN acts as a per-token memory/key-value store. Without it, the model relies entirely on attention for information routing. At this small scale (vocab=256, 9.6KB corpus), attention is sufficient for basic pattern matching. The FFN's value likely scales with task complexity — it may matter more on WikiText-2.

## Plot
`plots/no_ffn.png`
