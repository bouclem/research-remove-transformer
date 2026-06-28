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
- [x] **A1: Normalization removed** — all RMSNorm commented out, -1,248 params, PPL 215→255 (+18%). Model completely fails to learn — loss flat at random guessing. Normalization is essential.
- [x] **A2: Positional encoding removed** — RoPE commented out, 0 param change, PPL 215→225 (+4.5%). Causal mask provides implicit position info. RoPE helpful but not critical.
- [ ] **A3: Weight tying** — tied vs untied embedding/output
- [ ] **A4: Activation** — GELU vs ReLU vs SwiGLU
- [ ] **A5: Norm placement** — pre-norm vs post-norm
- [x] **A-FFN: Feed-forward removed** — 799K→246K params (-69%), PPL 215→235 (+9%). FFN not strictly necessary, attention alone learns structure.
- [ ] **A7: Head config** — fewer wide heads vs many narrow heads
- [ ] **A8: Bias terms** — no bias vs with bias on linear layers
- [ ] **A9: Attention scaling** — 1/sqrt(d) vs 1/d vs learned scale

## Infrastructure
- [ ] Experiment runner script (sweep configs, log results)
- [ ] Results comparison table/plot across ablations
- [ ] Config system for variant experiments
