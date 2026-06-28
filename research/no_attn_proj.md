# No Attention Output Projection

## What we did
Removed the `self.proj` linear layer from `CausalSelfAttention`. The concatenated multi-head attention output goes directly into the residual stream without a mixing projection.

## Architecture change
```python
# Before (baseline):
self.proj = nn.Linear(d_model, d_model, bias=False)
return self.proj(attn)

# After (no projection):
# self.proj = nn.Linear(d_model, d_model, bias=False)
return attn  # raw concatenated heads
```

## Results

| Metric | Baseline | No Attn Projection |
|---|---|---|
| Parameters | 799,968 | 744,192 (-55,776, -7%) |
| Best loss | 5.3731 | 5.4867 |
| Best PPL | 215.52 | 241.46 |
| Δ from baseline | — | +12.0% |
| Steps/s | 52.5 | 58.2 |

## Observations
- **Second-worst degradation** after no-norm (+12% PPL)
- Saves 7% of params but quality drops significantly
- Training is faster (58.2 vs 52.5 steps/s) — one fewer matmul per layer
- Loss plateaus high and never recovers

## Interpretation
The output projection serves a critical role: it mixes information across heads. Without it, the concatenated head outputs (each head_dim=16) are simply stacked and passed forward. The model has no way to combine findings from different heads — head 1's attention pattern can't inform head 3's representation.

This is essentially the "multi-head attention without output mixing" problem. The QKV projection creates diverse heads, but without the output projection, that diversity is wasted — the heads operate in isolation.

The output projection is parameter-efficient: 55K params (7% of model) for a 12% quality impact. It's one of the best param-to-impact ratios in the model.

## Plot
`plots/no_attn_proj.png`
