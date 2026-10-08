# mini-infer

A language model inference engine built from scratch in PyTorch, no `transformers` model code used at runtime, every component (attention, caching, memory management, GPU kernels) is hand-written and verified against the official Hugging Face implementation at every layer.

Base model: Qwen2.5-0.5B. Built and benchmarked on a 4GB RTX 3050 laptop GPU.

**Headline result: 3.55x faster than plain Hugging Face `generate()`** on identical hardware, identical weights, greedy decoding on both sides — 8 concurrent requests, continuous batching (`max_batch_size=8`) vs. running them one at a time. See [§7](#7-final-benchmark-vs-hugging-face-generate) for the full comparison.

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

### 4. Continuous batching
Serves multiple requests concurrently instead of one at a time. A scheduler admits waiting requests into a running batch, evicts finished ones, and immediately fills freed slots with new arrivals, all built on top of the paged cache (different sequences can have different lengths and still be processed together in one batched GPU call).

**Built and verified in layers:**
1. Scheduler and request lifecycle — 5 unit tests (batch-size enforcement, slot reuse on completion, prefill/decode transitions, EOS handling, liveness tracking)
2. Batched paged attention — pads variable-length sequences to the batch max and masks the padding with `-inf` before softmax; verified against running each sequence individually (max diff ~5e-7 to 8e-7, consistent with float32 rounding seen across the whole project)
3. Full 24-layer integration — scheduler-driven serving of 4 concurrent requests, with the batch size deliberately set smaller than the request count to force a slot to free up mid-run, produces output token-for-token identical to running each request alone

**Throughput, 8 requests × 30 tokens each, measured on the RTX 3050 (bfloat16):**

| Serving method | Time | Throughput | Speedup |
|---|---|---|---|
| Sequential (one at a time) | 9.75s | 24.6 tok/s | 1.00x |
| Continuous batching, max_batch=2 | 8.76s | 27.4 tok/s | 1.11x |
| Continuous batching, max_batch=4 | 5.49s | 43.7 tok/s | 1.77x |
| Continuous batching, max_batch=8 | 3.24s | 74.1 tok/s | **3.01x** |

Throughput scales with batch size because single-request decode is overhead-bound on this hardware (confirmed in the KV cache benchmarks), batching gives the GPU real parallel work per step instead of mostly idling between tiny sequential calls.

### 5. Custom Triton GPU kernels
Hand-written GPU kernels (not `torch` ops), compiled with Triton, each verified against the project's own proven PyTorch implementation before being benchmarked.

**RMSNorm kernel** — one GPU program per token/row, loading the row, computing the mean-square normalization and scaling by the learned weight entirely inside the kernel.
- Verified against both the project's own RMSNorm and the official HF layer directly: 1.19e-07 diff (float32 rounding floor) against both
- Benchmark (2000-token sequence): PyTorch 0.3015 ms/call vs. Triton 0.0827 ms/call — **3.65x faster**

**Paged-attention decode kernel** — the core systems idea of this project at the kernel level: reads Query/Key/Value directly out of the paged KV cache's physical memory pool (using `physical_page` indices and `kv_head_idx = head_idx // num_groups` for GQA head-sharing), with no Python-side gather or concatenation step before attention runs. Built and proven in two stages:
1. **Single-page case** (sequence fits in one page) — bit-exact match (0.00e+00) against `forward_paged`
2. **Multi-page case** (sequence spans many pages) — uses the *online softmax* technique (the same incremental running-max/running-sum/running-weighted-accumulator idea behind FlashAttention) to combine partial attention results page by page, without ever materializing the full K/V history as one tensor. The algorithm was proven correct in pure PyTorch first (`test_online_softmax_math.py`, matching a one-shot softmax to 1.19e-07) before being ported into the Triton kernel, where it matched `forward_paged` bit-exactly (0.00e+00) across a 37-token, 3-page sequence

**Benchmark (2000-token decode history, 126 pages, RTX 3050, float32):**

| Implementation | Time | Speedup |
|---|---|---|
| PyTorch `forward_paged` | 3.5462 ms/call | 1.00x |
| Triton paged-attention kernel | 0.1550 ms/call | **22.89x** |

The RMSNorm and paged-attention speedups differ by an order of magnitude for a reason: RMSNorm is a tiny per-row operation, so its runtime is dominated by fixed kernel-launch/memory overhead that both versions pay roughly equally, leaving a modest win. Paged-attention's win is much larger because `forward_paged`'s PyTorch path pays a real, growing cost — a Python-level loop in `PagedKVCache.read()` gathering across every page (126 of them here) into one contiguous tensor before attention can even start. The Triton kernel reads directly from scattered pages inside the GPU kernel itself, eliminating that gather step entirely — this is the actual mechanism, and the actual payoff, behind vLLM's real PagedAttention kernel.

### 6. Prefix caching
Lets requests that share an identical prompt prefix (the common case: a fixed system prompt in front of every user message) reuse the same physical pages instead of each one computing and storing its own copy. Only whole, fully-filled pages are ever shared — a page still being written into can't be handed to anyone else.

**Built and verified in layers:**
1. Reference-counted page ownership — extended `PageAllocator` so a physical page can be owned by more than one sequence at once; a page only returns to the free pool once every owner has released it. 11 unit tests (7 original + 4 new: shared-page ref counting, three-way sharing, attach-to-unregistered-sequence guard, mixed shared/private pages freeing independently)
2. `PrefixCache` — exact, page-aligned prefix matching: given a new request's prompt, finds the longest run of already-cached full pages it can reuse. 6 unit tests (empty cache, exact match, partial final page never shared, divergence stops the match at the right page, no-match case, longer prompt gets credit only for its shared portion)
3. End-to-end integration — two sequences sharing an 8-token prefix: the second reuses the first's pages and writes only its own new tokens, reads back a reconstruction that's bit-identical (0.0 diff) to the original, and — the critical safety check — freeing the *first* sequence does not corrupt or prematurely free the pages the *second* is still using
4. Memory impact: 200 requests sharing a 300-token system prompt (18 full pages), page size 16:
   - Without prefix caching: 4,738 total page-allocations
   - With prefix caching: 1,110 unique physical pages actually in use
   - **4.27x fewer physical pages needed**, from 3,582 shared-page reuse events (199 requests × 18 shared pages — exactly matching the hand-derived math)

### 7. Final benchmark vs. Hugging Face `generate()`
The number everything above was built toward: mini-infer's full serving stack (paged KV cache + continuous-batching scheduler) measured against plain `model.generate()` — no custom code at all — on the same GPU, same weights, same prompts, greedy decoding on both sides.

**8 requests, 30 new tokens each, RTX 3050, bfloat16:**

| Serving method | Time | Throughput | Speedup |
|---|---|---|---|
| Hugging Face `generate()` (sequential) | 11.80s | 20.3 tok/s | 1.00x |
| mini-infer, max_batch_size=2 | 9.36s | 25.6 tok/s | 1.26x |
| mini-infer, max_batch_size=4 | 5.76s | 41.7 tok/s | 2.05x |
| mini-infer, max_batch_size=8 | 3.32s | 72.2 tok/s | **3.55x** |

This is the full stack working together, not any one piece in isolation: the paged cache is what makes it safe to grow the batch size without reserving worst-case memory per request, and the scheduler is what turns that into fewer, larger, better-utilized GPU calls instead of one small call per request.

## Tech stack
Python, PyTorch, Triton (custom GPU kernels), Qwen2.5-0.5B weights via `transformers`/`safetensors` (loading only, not inference), WSL2 + CUDA.

## Hardware
Developed and benchmarked on a 4GB RTX 3050 laptop GPU, 16GB RAM. Everything here runs on free, local compute, no cloud spend.

## Project structure

```
model/
  rmsnorm.py                 RMSNorm, verified
  rmsnorm_triton.py          Triton RMSNorm kernel, verified
  rope.py                    RoPE: single-sequence and batched variants, verified
  attention.py                Grouped-query attention: contiguous, paged, and batched-paged, verified
  feedforward.py               SwiGLU feed-forward block, verified
  layer.py                    One transformer layer (contiguous, paged, batched-paged)
  full_model.py                Full 24-layer model (contiguous, paged, batched-paged)
  page_allocator.py            Page allocation and tracking, with reference counting for shared pages
  paged_cache.py                Paged KV storage (write/read)
  scheduler.py                  Request lifecycle and continuous-batching scheduler
  paged_attention_triton.py     Triton paged-attention decode kernels (single-page and multi-page/online-softmax), verified
  prefix_cache.py                Exact, page-aligned prefix matching for shared prompt prefixes, verified

test_rmsnorm.py                         RMSNorm correctness
test_rope.py                             RoPE correctness
test_attention.py                        Attention correctness (contiguous)
test_feedforward.py                      Feed-forward correctness
test_full_model.py                       Full model correctness (no cache)
test_kv_cache.py                         KV cache correctness + speedup
test_page_allocator.py                   Page allocator unit tests
test_paged_cache.py                      Paged storage unit tests
test_paged_attention.py                  Paged attention vs. contiguous cache
test_full_paged_model.py                 Full model with paged cache, end to end
test_scheduler.py                        Scheduler unit tests
test_batched_attention.py                Batched paged attention vs. per-sequence
test_continuous_batching.py              Full continuous batching, end to end
test_triton_setup.py                     Triton sanity check (vector add)
test_triton_rmsnorm.py                   Triton RMSNorm vs. project RMSNorm and HF
test_online_softmax_math.py              Online-softmax algorithm proof (pure PyTorch, pre-Triton)
test_triton_paged_attention.py           Triton paged-attention kernel, single-page case
test_triton_paged_attention_multipage.py Triton paged-attention kernel, multi-page/online-softmax case
test_prefix_cache.py                     PrefixCache unit tests (matching, divergence, partial pages)
test_prefix_cache_integration.py         Prefix caching end to end: sharing, correctness, safe freeing

bench_gpu.py                     GPU timing + memory-per-token benchmark
bench_paged_memory.py            Paged vs. naive memory, concurrent request capacity
bench_continuous_batching.py     Throughput: sequential vs. continuous batching
bench_triton_rmsnorm.py          Triton vs. PyTorch RMSNorm speed
bench_triton_paged_attention.py  Triton vs. PyTorch paged-attention decode speed
bench_prefix_cache.py            Prefix caching memory savings, shared system prompt scenario
bench_vs_huggingface.py          Final benchmark: mini-infer vs. plain HF generate()
check_match.py                    float32 vs. bfloat16 cache divergence check
inspect_config.py                 Prints the model's config for reference
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
python3 test_triton_setup.py
python3 test_triton_rmsnorm.py
python3 test_online_softmax_math.py
python3 test_triton_paged_attention.py
python3 test_triton_paged_attention_multipage.py
python3 test_prefix_cache.py
python3 test_prefix_cache_integration.py
```

## Roadmap
- [x] Model built from scratch, verified against official weights
- [x] KV cache (prefill/decode split)
- [x] Paged KV cache
- [x] Continuous batching
- [x] Custom Triton attention kernel (RMSNorm + single-page and multi-page paged-attention decode)
- [x] Prefix caching
- [x] Final throughput/latency benchmark vs. Hugging Face `generate` (**3.55x**)
- [ ] Speculative decoding
