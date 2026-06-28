# No FFN + No RoPE (Combined Ablation)

## What we did
Commented out both the FFN and RoPE. The model is attention-only with normalization but no positional encoding.

## Architecture change
```python
# Block forward (no FFN, no RoPE):
x = x + self.attn(self.norm1(x))
# No FFN, no RoPE on Q/K
```

## Results

| Metric | Baseline | No FFN | No RoPE | No FFN + No RoPE |
|---|---|---|---|---|
| Params | 799,968 | 246,432 | 799,968 | 246,432 |
| Best PPL | 215.52 | 234.84 | 225.24 | 235.43 |
| Δ from baseline | — | +8.9% | +4.5% | +9.2% |
| Steps/s | 52.5 | 67.6 | 61.0 | 61.0 |

## Observations
- Combined PPL (235.43) ≈ no-FFN alone (234.84) — RoPE removal adds only +0.3%
- The degradations are NOT additive: +8.9% + +4.5% would be ~+13.4%, but we see only +9.2%
- FFN removal dominates — without per-token transformation, positional info matters less
- The model still learns (unlike no-norm) — normalization keeps it stable

## Interpretation
FFN and RoPE interact. When the FFN is present, it transforms attention output using position-aware representations, so RoPE's positional signal is utilized. Without FFN, attention just routes raw embeddings — positional distinctions in Q/K matter less because there's no downstream transformation that uses them.

This suggests component ablations can be non-additive: removing two components together may be less harmful than the sum of removing each individually.

## Plot
`plots/no_ffn_no_rope.png`
