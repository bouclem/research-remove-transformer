# A4: SwiGLU Activation

## What we did
Replaced the standard FFN (`up -> GELU -> down`) with SwiGLU (`Swish(gate(x)) * up(x) -> down`). This adds a third linear layer (gate) and uses element-wise multiplication for gating.

## Architecture change
```python
# Before (baseline GELU):
self.up = nn.Linear(d_model, d_ff, bias=False)
self.down = nn.Linear(d_ff, d_model, bias=False)
return self.down(F.gelu(self.up(x)))

# After (SwiGLU):
self.gate = nn.Linear(d_model, d_ff, bias=False)  # extra layer
self.up = nn.Linear(d_model, d_ff, bias=False)
self.down = nn.Linear(d_ff, d_model, bias=False)
return self.down(F.silu(self.gate(x)) * self.up(x))
```

## Results

| Metric | GELU (baseline) | ReLU | SwiGLU |
|---|---|---|---|
| Parameters | 799,968 | 799,968 | 1,076,448 (+35%) |
| Best loss | 5.3731 | 5.3625 | 5.4032 |
| Best PPL | 215.52 | 213.25 | 222.11 |
| Δ from baseline | — | -1.1% | +3.0% |
| Steps/s | 52.5 | 56.2 | 50.1 |

## Observations
- **SwiGLU is worse** despite 35% more parameters — PPL 222 vs 215
- Training is slower (50.1 vs 52.5 steps/s) — extra gate projection + element-wise multiply
- Loss converges to a higher floor than both GELU and ReLU
- The gating mechanism doesn't help at this scale

## Interpretation
SwiGLU was designed for large-scale models (LLaMA, PaLM) where the gating helps with capacity utilization. At 800K params with 9.6KB of training data, the gate adds parameters that don't get enough training signal to learn useful gating patterns. The extra capacity is wasted.

Combined with the ReLU result, the activation hierarchy at small scale is:
**ReLU > GELU > SwiGLU**

Simpler is better. The sophisticated activations that help at scale become overhead at small scale.

## Plot
`plots/swiglu.png`
