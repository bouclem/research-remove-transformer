# Changelog

## [Unreleased]

## [0.1.0] - 2026-06-28
### Added
- 800K parameter decoder-only transformer baseline (799,968 params)
  - d_model=96, n_layers=6, n_heads=6, d_ff=480, vocab=256
  - RMSNorm (pre-norm, no bias), RoPE, GELU FFN, weight tying
- Byte-level tokenizer (0-255, UTF-8)
- Step-based training loop with AdamW, cosine LR + warmup, mixed precision, gradient clipping
- Dataset system: custom corpus + WikiText-2 (HuggingFace datasets)
- Matplotlib training curve plotting (loss + perplexity)
- Component folder structure: model/, tokenizer/, datasets/, plots/
- .gitignore, docs/README.md, docs/TODO.md
