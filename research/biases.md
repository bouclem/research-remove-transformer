# Biases on Linear Layers

## What we did
Added `bias=True` to all linear layers (QKV, attention output projection, FFN up/down). Biases initialized to zero.

## Architecture change
```python
# Before (baseline — no biases):
self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
self.proj = nn.Linear(d_model, d_model, bias=False)
self.up = nn.Linear(d_model, d_ff, bias=False)
self.down = nn.Linear(d_ff, d_model, bias=False)

# After (with biases):
self.qkv = nn.Linear(d_model, 3 * d_model, bias=True)
self.proj = nn.Linear(d_model, d_model, bias=True)
self.up = nn.Linear(d_model, d_ff, bias=True)
self.down = nn.Linear(d_ff, d_model, bias=True)
```

## Results

| Metric | Baseline (no bias) | With Biases |
|---|---|---|
| Parameters | 799,968 | 814,464 (+14,496) |
| Best loss | 5.3731 | 5.4513 |
| Best PPL | 215.52 | 233.07 |
| Δ from baseline | — | +8.1% (worse) |
| Steps/s | 52.5 | 50.8 |

## Observations
- **Biases significantly hurt** — +8.1% PPL despite +14K params
- Training is slightly slower (50.8 vs 52.5 steps/s) — extra add operations
- Loss converges to a much higher floor
- The degradation is consistent throughout training, not just at convergence

## Interpretation
The model uses RMSNorm which already provides per-dimension scaling. Adding biases to linear layers creates redundant degrees of freedom — the model has to learn both the bias terms AND the norm parameters that serve similar purposes (shifting/scaling activations).

With limited training data (9.6KB) and steps (2000), the extra parameters dilute optimization signal. The optimizer spends capacity learning biases that don't contribute useful representational power.

This explains why modern architectures (LLaMA, Mistral) drop biases: with normalization handling the shifting, biases are pure overhead. They add params, memory, compute, and optimization difficulty without benefit.

## Plot
`plots/biases.png`
