# mini-infer

A language model inference engine built from scratch in PyTorch, no `transformers` model code used at runtime, every component (attention, caching, memory management) is hand-written and verified against the official Hugging Face implementation at every layer.

Base model: Qwen2.5-0.5B. Built and benchmarked on a 4GB RTX 3050 laptop GPU.

## Why this project exists

Most ML portfolios stop at "I can call a model." This project goes one level deeper: it rebuilds the systems that make serving a language model fast and memory-efficient, the same ideas used in production engines like vLLM, at small scale and with full correctness proofs at every step.

## What's implemented

### 1. The model, from scratch

Every component of the transformer is hand-written and unit-tested against Hugging Face's official Qwen2.5-0.5B:

| Component               | What it does                                       | Verified accuracy                                                    |
| ----------------------- | -------------------------------------------------- | -------------------------------------------------------------------- |
| RMSNorm                 | Keeps activations numerically stable across layers | Exact match (0.0 diff)                                               |
| RoPE                    | Encodes token position via rotation                | Exact match (0.0 diff)                                               |
| Grouped-query attention | 14 query heads sharing 2 KV head groups            | Match (1.19e-07, float32 rounding floor)                             |
| SwiGLU feed-forward     | Gated per-token transformation                     | Exact match (0.0 diff)                                               |
| Full 24-layer model     | All components assembled with residual connections | Logits match (2e-05); generated text identical to the official model |

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

### 4. Continuous batching

Serves multiple requests concurrently instead of one at a time. A scheduler admits waiting requests into a running batch, evicts finished ones, and immediately fills freed slots with new arrivals, all built on top of the paged cache (different sequences can have different lengths and still be processed together in one batched GPU call).

**Built and verified in layers:**

1. Scheduler and request lifecycle — 5 unit tests (batch-size enforcement, slot reuse on completion, prefill/decode transitions, EOS handling, liveness tracking)
2. Batched paged attention — pads variable-length sequences to the batch max and masks the padding with `-inf` before softmax; verified against running each sequence individually (max diff ~5e-7 to 8e-7, consistent with float32 rounding seen across the whole project)
3. Full 24-layer integration — scheduler-driven serving of 4 concurrent requests, with the batch size deliberately set smaller than the request count to force a slot to free up mid-run, produces output token-for-token identical to running each request alone

**Throughput, 8 requests × 30 tokens each, measured on the RTX 3050 (bfloat16):**

| Serving method                   | Time  | Throughput | Speedup   |
| -------------------------------- | ----- | ---------- | --------- |
| Sequential (one at a time)       | 9.75s | 24.6 tok/s | 1.00x     |
| Continuous batching, max_batch=2 | 8.76s | 27.4 tok/s | 1.11x     |
| Continuous batching, max_batch=4 | 5.49s | 43.7 tok/s | 1.77x     |
| Continuous batching, max_batch=8 | 3.24s | 74.1 tok/s | **3.01x** |

Throughput scales with batch size because single-request decode is overhead-bound on this hardware (confirmed in the KV cache benchmarks), batching gives the GPU real parallel work per step instead of mostly idling between tiny sequential calls.

## Tech stack

Python, PyTorch, Triton (planned), Qwen2.5-0.5B weights via `transformers`/`safetensors` (loading only, not inference), WSL2 + CUDA.

## Hardware

Developed and benchmarked on a 4GB RTX 3050 laptop GPU, 16GB RAM. Everything here runs on free, local compute, no cloud spend.

## Project structure

```
model/
  rmsnorm.py            RMSNorm, verified
  rope.py                RoPE: single-sequence and batched variants, verified
  attention.py           Grouped-query attention: contiguous, paged, and batched-paged, verified
  feedforward.py          SwiGLU feed-forward block, verified
  layer.py               One transformer layer (contiguous, paged, batched-paged)
  full_model.py           Full 24-layer model (contiguous, paged, batched-paged)
  page_allocator.py       Page allocation and tracking
  paged_cache.py          Paged KV storage (write/read)
  scheduler.py            Request lifecycle and continuous-batching scheduler

test_rmsnorm.py            RMSNorm correctness
test_rope.py                RoPE correctness
test_attention.py           Attention correctness (contiguous)
test_feedforward.py         Feed-forward correctness
test_full_model.py          Full model correctness (no cache)
test_kv_cache.py            KV cache correctness + speedup
test_page_allocator.py      Page allocator unit tests
test_paged_cache.py         Paged storage unit tests
test_paged_attention.py     Paged attention vs. contiguous cache
test_full_paged_model.py    Full model with paged cache, end to end
test_scheduler.py           Scheduler unit tests
test_batched_attention.py   Batched paged attention vs. per-sequence
test_continuous_batching.py Full continuous batching, end to end

bench_gpu.py                 GPU timing + memory-per-token benchmark
bench_paged_memory.py        Paged vs. naive memory, concurrent request capacity
bench_continuous_batching.py Throughput: sequential vs. continuous batching
check_match.py                float32 vs. bfloat16 cache divergence check
inspect_config.py             Prints the model's config for reference
```

## Running the tests

```bash
source venv/bin/activate
python3 test_rmsnorm.py
python3 test_rope.py
python3 test_attention.py
python3 test_feedforward.py
python3 test_full_model.py
python3 test_kv_cache.py
python3 test_page_allocator.py
python3 test_paged_cache.py
python3 test_paged_attention.py
python3 test_full_paged_model.py
python3 test_scheduler.py
python3 test_batched_attention.py
python3 test_continuous_batching.py
```

## Roadmap

- [x] Model built from scratch, verified against official weights
- [x] KV cache (prefill/decode split)
- [x] Paged KV cache
- [x] Continuous batching
- [ ] Custom Triton attention kernel
- [ ] Prefix caching or speculative decoding
- [ ] Final throughput/latency benchmarks vs. Hugging Face `generate`
