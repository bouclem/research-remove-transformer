# ALiBi vs RoPE

## What we did
Replaced RoPE with ALiBi (Attention with Linear Biases). ALiBi adds a per-head linear distance penalty to attention scores: `-m_h * (i - j)` where `m_h = 1/2^(8h/n_heads)`. No positional information is added to the input embeddings.

## Architecture change
```python
# Before (baseline — RoPE):
q = apply_rope(q, self.rope_cos, self.rope_sin)
k = apply_rope(k, self.rope_cos, self.rope_sin)
attn = F.scaled_dot_product_attention(q, k, v, is_causal=True)

# After (ALiBi):
# No RoPE applied to Q, K
slopes = 1.0 / (2 ** (8 * torch.arange(n_heads) / n_heads))
dist = positions[:, None] - positions[None, :]  # (T, T)
alibi_bias = -slopes[:, None, None] * dist[None, :, :].float()
alibi_bias = alibi_bias.masked_fill(dist[None, :, :] > 0, float('-inf'))
attn = F.scaled_dot_product_attention(q, k, v, attn_mask=alibi_bias, is_causal=False)
```

## Results

| Metric | Baseline (RoPE) | No RoPE | Learned Pos | ALiBi |
|---|---|---|---|---|
| Parameters | 799,968 | 799,968 | 831,488 | 799,968 (same) |
| Best PPL | 215.52 | 225.24 | 234.48 | 219.63 |
| Δ from baseline | — | +4.5% | +8.8% | +1.9% |
| Steps/s | 52.5 | 56.1 | 60.5 | 62.6 |

## Observations
- **ALiBi is the best RoPE alternative** — only +1.9% PPL vs RoPE
- Same param count as RoPE (0 extra params) — pure mathematical bias
- Fastest training of all positional methods (62.6 steps/s)
- Loss is noisier than RoPE but converges to a reasonable floor
- Significantly better than no positional encoding (+1.9% vs +4.5%)

## Interpretation
ALiBi provides positional information by penalizing attention to distant tokens. Each head gets a different slope, creating a hierarchy: some heads attend locally (steep slope), others globally (shallow slope). This is a strong inductive bias for language — nearby tokens are usually more relevant.

Compared to RoPE:
- **RoPE** rotates Q and K in the attention space, encoding relative positions multiplicatively. This preserves the dot product's structure and allows precise positional reasoning.
- **ALiBi** adds a simple additive linear penalty. It's less expressive but provides a clean locality prior.

ALiBi's key advantage is extrapolation: it generalizes to longer sequences than seen in training (the linear bias extends naturally). RoPE also extrapolates but can degrade. At seq_len=128 with 2000 steps, this advantage isn't tested, but ALiBi is nearly as good while being simpler.

Positional encoding hierarchy: **RoPE > ALiBi > No pos > Learned pos**

## Plot
`plots/alibi.png`
