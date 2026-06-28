# Combo: ReLU + No Scaling

## What we did
Combined the two individual improvements (ReLU and no attention scaling) into one model.

## Results

| Metric | Baseline | ReLU only | No Scaling only | ReLU + No Scaling |
|---|---|---|---|---|
| PPL | 215.52 | 213.25 | 212.36 | 222.85 |
| Δ from baseline | — | -1.1% | -1.5% | +3.4% (worse) |

## Observations
- **The improvements cancel out and then some** — individually they help, together they hurt
- Expected additive effect: -1.1% + -1.5% = -2.6% improvement
- Actual combined effect: +3.4% degradation
- This is a -6.0% swing from expected — a strong negative interaction

## Interpretation
Both ReLU and no-scaling independently create "sharper" signal processing:
- ReLU has harder thresholds than GELU (zero or pass, no smooth transition)
- No scaling produces sharper softmax (larger logits → more peaked distributions)

Individually, this sharpness helps the model make decisive decisions. But combined, the sharpness compounds:
1. ReLU creates sparse, high-magnitude FFN activations
2. These feed into the next layer's QKV projection
3. Without scaling, the already-large QK dot products become even larger
4. Softmax saturates → attention collapses to near-one-hot distributions
5. Gradient flow degrades → learning stalls

This is a **sharpness trap**: moderate sharpness improves signal, but excessive sharpness destroys gradient flow. The two improvements operate on the same axis (activation sharpness) and overshoot when combined.

## Plot
`plots/relu_no_scaling.png`
