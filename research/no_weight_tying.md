# A3: Weight Tying Removed

## What we did
Added a separate `lm_head` linear layer instead of using the token embedding weights for output projection.

## Architecture change
```python
# Before (baseline — weight tying):
logits = x @ self.token_embedding.weight.T

# After (untied):
self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
logits = self.lm_head(x)
```

## Results

| Metric | Baseline (tied) | No Weight Tying |
|---|---|---|
| Parameters | 799,968 | 824,544 (+24,576) |
| Best loss | 5.3731 | 5.4401 |
| Best PPL | 215.52 | 230.47 |
| Δ from baseline | — | +6.9% |
| Steps/s | 52.5 | 55.9 |

## Observations
- **Weight tying is very helpful** — removing it costs +6.9% PPL despite +3% more params
- The extra `lm_head` params (24,576) don't compensate — they need to learn the token space from scratch
- Training is slightly faster (55.9 vs 52.5 steps/s) — separate matmul vs transposed embedding lookup
- Loss converges to a clearly worse floor

## Interpretation
Weight tying provides a strong inductive bias: the space that represents input tokens is the same space used to predict them. This creates a circular consistency — tokens that are similar in the input space are also similar in the output space. Without tying, the model must learn this alignment from limited data.

At small scale with limited data (9.6KB), this inductive bias is critical. The 24K extra params in `lm_head` are wasted — they can't learn a better mapping than the one the embedding already provides.

Weight tying is also parameter-efficient: it saves 24,576 params while improving quality. It's one of the few "free lunches" in transformer design.

## Plot
`plots/no_weight_tying.png`
