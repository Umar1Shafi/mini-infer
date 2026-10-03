# mini-infer

A language model inference engine built from scratch in PyTorch, no `transformers` model code used at runtime, every component (attention, caching, memory management) is hand-written and verified against the official Hugging Face implementation at every layer.

Base model: Qwen2.5-0.5B. Built and benchmarked on a 4GB RTX 3050 laptop GPU.

## Why this project exists

Most ML portfolios stop at "I can call a model." This project goes one level deeper: it rebuilds the systems that make serving a language model fast and memory-efficient, the same ideas used in production engines like vLLM, at small scale and with full correctness proofs at every step.

## What's implemented

### 1. The model, from scratch
Every component of the transformer is hand-written and unit-tested against Hugging Face's official Qwen2.5-0.5B:

| Component | What it does | Verified accuracy |
|---|---|---|
| RMSNorm | Keeps activations numerically stable across layers | Exact match (0.0 diff) |
| RoPE | Encodes token position via rotation | Exact match (0.0 diff) |
| Grouped-query attention | 14 query heads sharing 2 KV head groups | Match (1.19e-07, float32 rounding floor) |
| SwiGLU feed-forward | Gated per-token transformation | Exact match (0.0 diff) |
| Full 24-layer model | All components assembled with residual connections | Logits match (2e-05); generated text identical to the official model |

### 2. KV cache
Splits generation into prefill (process the prompt once) and decode (generate one token at a time, reusing cached Keys/Values instead of recomputing the whole sequence).

- Verified bit-identical to no-cache generation (float32, both CPU and GPU)
- bfloat16 diverges only after ~20 generated tokens, consistent with expected low-precision rounding, not a logic bug
- Confirmed cache memory cost: **12,288 bytes per token** (24 layers × 2 [K,V] × 2 KV heads × 64 dims × 2 bytes), matching the hand-derived formula exactly

### 3. Paged KV cache (the core systems contribution)
Replaces per-request contiguous memory reservation with a vLLM-style paged allocator: memory is split into fixed-size pages (16 tokens each), handed out to sequences on demand from a shared pool, and returned when a sequence finishes.

**Built and verified in layers, each proven before the next was built on top of it:**
1. Page allocator — 7 unit tests (allocation, exact boundary math, incremental growth, out-of-memory handling, cross-sequence isolation, position→page mapping)
2. Page-based storage — 5 unit tests (round-trips, page-boundary crossings, incremental decode writes, cross-sequence isolation, cross-layer isolation)
3. Paged attention — bit-exact match (0.00e+00) against the existing contiguous-cache attention across 11 generation steps, including a deliberately forced page-boundary crossing
4. Full 24-layer integration — exact token-ID match against contiguous-cache generation
5. Memory impact, measured two independent ways:
   - Analytical: **806 vs. 7** max concurrent requests in the same memory budget (**~115x**), assuming realistic 300-token average conversations
   - Live simulation: 200 randomly-sized conversations (20-800 tokens) all served using only 34.6% of the memory budget that would be needed for just 7 naive requests

## Tech stack
Python, PyTorch, Triton (planned), Qwen2.5-0.5B weights via `transformers`/`safetensors` (loading only, not inference), WSL2 + CUDA.

## Hardware
Developed and benchmarked on a 4GB RTX 3050 laptop GPU, 16GB RAM. Everything here runs on free, local compute, no cloud spend.

## Project structure
