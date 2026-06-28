# Research Findings

Preliminary findings from the transformer ablation study. These are not final — they're observations from individual experiments at small scale (800K params, 9.6KB corpus, 2000 steps).

## Summary Table

| Experiment | Params | PPL | Δ from baseline | Plot |
|---|---|---|---|---|
| Baseline | 799,968 | 215.52 | — | `plots/baseline.png` |
| No FFN | 246,432 | 234.84 | +8.9% | `plots/no_ffn.png` |
| No RoPE | 799,968 | 225.24 | +4.5% | `plots/no_rope.png` |

## Key Findings So Far

### 1. The FFN is expensive but not essential
The FFN is 69% of parameters but removing it only costs 9% PPL. Attention alone can learn basic language structure. The FFN may be a luxury that scales in importance with task complexity.

### 2. Positional encoding is helpful but not required
RoPE costs 0 parameters and only 4.5% PPL when removed. The causal mask provides implicit positional signal. This challenges the conventional wisdom that positional encoding is strictly necessary.

### 3. Both removals speed up training
No-FFN: +29% steps/s. No-RoPE: +16% steps/s. Fewer operations = faster training, which matters for research iteration speed.

## Caveats
- Small corpus (9.6KB) — patterns may be too simple to stress-test components
- Short sequences (128) — positional encoding may matter more at longer lengths
- Only 2000 steps — some components might show larger gaps with more training
- No eval split — all numbers are training loss, not generalization

## Next Questions
- Does the FFN matter more on WikiText-2 (larger, more complex)?
- Does RoPE matter more at seq_len=256 or 512?
- What happens if we remove BOTH FFN and RoPE?
- What about removing normalization (A1)?
