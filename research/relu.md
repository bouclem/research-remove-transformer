# A4: GELU → ReLU

## What we did
Swapped the activation function in the FFN from GELU to ReLU.

## Architecture change
```python
# Before (baseline):
return self.down(F.gelu(self.up(x)))

# After (ReLU):
return self.down(F.relu(self.up(x)))
```

## Results

| Metric | Baseline (GELU) | ReLU |
|---|---|---|
| Parameters | 799,968 | 799,968 (same) |
| Best loss | 5.3731 | 5.3625 |
| Best PPL | 215.52 | 213.25 |
| Δ from baseline | — | -1.1% (better) |
| Steps/s | 52.5 | 56.2 |

## Observations
- **ReLU is slightly better than GELU** — PPL 213.25 vs 215.52
- Training is faster (+7% steps/s) — ReLU is computationally simpler
- Loss curve is similar shape, slightly lower throughout
- No instability or dead neuron issues observed

## Interpretation
GELU was introduced to smooth ReLU's gradient and improve performance in larger models. At 800K params with a small corpus, this smoothing provides no benefit — ReLU's simplicity wins. The extra computation of GELU (involving erf/sigmoid) is pure overhead at this scale.

This suggests GELU's advantage is scale-dependent — it likely matters more in larger models where smooth gradients help optimization navigate complex loss landscapes. At small scale, ReLU's harder thresholding may actually help with feature sparsity.

## Plot
`plots/relu.png`
