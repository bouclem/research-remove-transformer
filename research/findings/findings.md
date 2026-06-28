# Research Findings

Preliminary findings from the transformer ablation study. These are not final — they're observations from individual experiments at small scale (800K params, 9.6KB corpus, 2000 steps).

## Summary Table

| Experiment | Params | PPL | Δ from baseline | Plot |
|---|---|---|---|---|
| Baseline | 799,968 | 215.52 | — | `plots/baseline.png` |
| No FFN | 246,432 | 234.84 | +8.9% | `plots/no_ffn.png` |
| No RoPE | 799,968 | 225.24 | +4.5% | `plots/no_rope.png` |
| No Norm | 798,720 | 254.93 | +18.3% | `plots/no_norm.png` |
| No FFN + No RoPE | 246,432 | 235.43 | +9.2% | `plots/no_ffn_no_rope.png` |
| No Residual | 799,968 | 217.33 | +0.8% | `plots/no_residual.png` |
| ReLU (vs GELU) | 799,968 | 213.25 | -1.1% (better) | `plots/relu.png` |
| SwiGLU (vs GELU) | 1,076,448 | 222.11 | +3.0% | `plots/swiglu.png` |

## Key Findings So Far

### 1. Normalization is essential — the model fails without it
Removing RMSNorm (only 1,248 params = 0.16% of model) causes total failure. Loss stays flat at random guessing (PPL 255 ≈ ln(256)). This is the most critical component — almost free in params but completely necessary for learning. Residual connections without norm cause activation explosion across layers.

### 2. The FFN is expensive but not essential
The FFN is 69% of parameters but removing it only costs 9% PPL. Attention alone can learn basic language structure. The FFN may be a luxury that scales in importance with task complexity.

### 3. Positional encoding is helpful but not required
RoPE costs 0 parameters and only 4.5% PPL when removed. The causal mask provides implicit positional signal. This challenges the conventional wisdom that positional encoding is strictly necessary.

### 4. Component criticality ranking
1. **Normalization** — essential (removal = total failure, -0.16% params)
2. **FFN** — useful but not critical (removal = -69% params, +9% PPL)
3. **RoPE** — nice to have (removal = 0 params, +4.5% PPL)
4. **Residual connections** — barely matters at 6 layers (removal = 0 params, +0.8% PPL, noisier training)

### 5. All removals speed up training
No-Norm: +27% steps/s. No-FFN: +29% steps/s. No-RoPE: +16% steps/s. No-Residual: +3% steps/s. But only FFN removal is worth the quality trade-off.

### 6. Component ablations are non-additive
Removing FFN alone: +8.9% PPL. Removing RoPE alone: +4.5% PPL. Removing both: +9.2% PPL (not +13.4%). FFN and RoPE interact — without FFN, positional encoding matters less because there's no per-token transformation to utilize position-aware representations.

### 7. Residual connections matter less at shallow depth
At 6 layers, removing residuals costs only +0.8% PPL. The signal flows fine through sequential attention+FFN. However, training is noisier (larger loss swings), suggesting residuals help optimization stability even when they don't affect final quality. This likely reverses at greater depth (24+ layers).

### 8. Simpler activations win at small scale
Activation hierarchy: **ReLU > GELU > SwiGLU**. ReLU improves PPL by 1.1% over GELU and is 7% faster. SwiGLU is 3% worse than GELU despite 35% more params and is slower. GELU's smooth gradient and SwiGLU's gating mechanism are designed for large-scale optimization — at 800K params, they're overhead without benefit.

## Caveats
- Small corpus (9.6KB) — patterns may be too simple to stress-test components
- Short sequences (128) — positional encoding may matter more at longer lengths
- Only 2000 steps — some components might show larger gaps with more training
- No eval split — all numbers are training loss, not generalization

## Next Questions
- Does the FFN matter more on WikiText-2 (larger, more complex)?
- Does RoPE matter more at seq_len=256 or 512?
- Do residuals matter more with more layers (12, 24)?
- Does GELU beat ReLU at larger scale or longer training?
- What about removing weight tying?
- What about SwiGLU?
