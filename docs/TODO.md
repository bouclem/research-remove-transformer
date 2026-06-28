# Research TODO

## Research Question
What do transformers need and don't need, to be more optimized but still basically the same?

## Baseline (DONE)
- [x] 800K param decoder-only transformer (799,968 params)
- [x] Byte-level tokenizer (vocab=256)
- [x] Step-based training loop (AdamW, cosine LR, mixed precision)
- [x] Datasets: custom corpus + WikiText-2
- [x] Training curve plotting (matplotlib)

## Ablation Experiments
- [ ] **A1: Normalization** — RMSNorm vs LayerNorm vs no norm
- [ ] **A2: Positional encoding** — RoPE vs learned PE vs no PE
- [ ] **A3: Weight tying** — tied vs untied embedding/output
- [ ] **A4: Activation** — GELU vs ReLU vs SwiGLU
- [ ] **A5: Norm placement** — pre-norm vs post-norm
- [ ] **A6: FFN ratio** — d_ff/d_model = 3x vs 4x vs 2x
- [ ] **A7: Head config** — fewer wide heads vs many narrow heads
- [ ] **A8: Bias terms** — no bias vs with bias on linear layers
- [ ] **A9: Attention scaling** — 1/sqrt(d) vs 1/d vs learned scale

## Infrastructure
- [ ] Experiment runner script (sweep configs, log results)
- [ ] Results comparison table/plot across ablations
- [ ] Config system for variant experiments
