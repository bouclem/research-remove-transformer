# 800K Parameter Transformer

A real, trainable decoder-only transformer language model with **799,968 parameters** (~800K).

## Architecture

| Component | Value |
|---|---|
| Type | Decoder-only Transformer |
| Parameters | **799,968** |
| vocab_size | 256 (byte-level) |
| d_model | 96 |
| n_layers | 6 |
| n_heads | 6 (head_dim=16) |
| d_ff | 480 |
| max_seq_len | 256 |
| Positional encoding | RoPE (0 params) |
| Normalization | RMSNorm (pre-norm, no bias) |
| Activation | GELU |
| Weight tying | Yes (embedding = output projection) |
| Linear biases | No |

### Parameter Breakdown

| Component | Params |
|---|---|
| Token embedding (256×96) | 24,576 |
| Attention (6 layers × 4×96²) | 221,184 |
| FFN (6 layers × 2×96×480) | 552,960 |
| RMSNorm (6×2 + 1) × 96 | 1,248 |
| **Total** | **799,968** |

## Project Structure

```
4/
├── model/
│   ├── __init__.py
│   └── transformer.py     — Transformer (RMSNorm, RoPE, attention, FFN)
├── tokenizer/
│   ├── __init__.py
│   └── byte_tokenizer.py  — Byte-level tokenizer (0-255)
├── datasets/
│   ├── __init__.py
│   ├── dataset.py         — Dataset loading (custom + WikiText-2)
│   └── sample_corpus.txt  — Custom proverbs training data
├── plots/
│   ├── __init__.py
│   └── plotter.py         — Matplotlib training curve plotting
├── train.py               — Step-based training script
├── generate.py            — Text generation (temperature, top-k)
├── requirements.txt       — Dependencies
└── docs/
    └── README.md          — This file
```

## Datasets

| Name | Source | Description |
|---|---|---|
| `custom` | `datasets/sample_corpus.txt` | Custom proverbs corpus (local file) |
| `wikitext2` | HuggingFace `datasets` | WikiText-2 raw (auto-download) |

## Quick Start

```bash
pip install -r requirements.txt

# Verify model architecture and param count
python -m model.transformer

# Train on custom corpus (step-based, no epochs)
python train.py --dataset custom --steps 5000 --batch_size 16

# Train on WikiText-2
python train.py --dataset wikitext2 --steps 10000 --device cuda

# Generate text
python generate.py --checkpoint checkpoints/best.pt --prompt "The" --max_tokens 200 --temperature 0.8
```

## Training Features

- **Step-based** training (no epoch concept, infinite data loader)
- **Dataset selection** — `--dataset custom` or `--dataset wikitext2`
- **AdamW** optimizer with weight decay (decay only on 2D+ params)
- **Cosine LR schedule** with linear warmup
- **Mixed precision** (bfloat16 on CUDA)
- **Gradient clipping** (max norm 1.0)
- **Weight tying** (embedding ↔ output projection)
- **Periodic evaluation** on validation split (when available)
- **Checkpointing** (best + periodic saves by step)
- **Training curves** — automatic matplotlib plot saved to `plots/`
- **CPU and CUDA** support

## Design Notes

- **RoPE** (Rotary Position Embedding) encodes positions with zero parameters
- **RMSNorm** replaces LayerNorm (removes bias, reduces params)
- **No linear biases** following modern transformer conventions (LLaMA-style)
- **Byte-level tokenizer** — no training needed, handles any UTF-8 text
- **Weight tying** reduces params by sharing embedding and output projection
- **WikiText-2** auto-downloads via HuggingFace `datasets` library
