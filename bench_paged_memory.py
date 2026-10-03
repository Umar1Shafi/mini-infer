"""
Memory comparison: naive (contiguous, worst-case reservation) vs paged KV cache,
using this project's real config numbers (Qwen2.5-0.5B on a 4GB GPU).
"""
from model.page_allocator import PageAllocator

# ---- Real numbers from this project ----
NUM_LAYERS = 24
NUM_KV_HEADS = 2
HEAD_DIM = 64
BYTES_PER_ELEM = 2  # bfloat16
BYTES_PER_TOKEN = NUM_LAYERS * 2 * NUM_KV_HEADS * HEAD_DIM * BYTES_PER_ELEM  # confirmed = 12,288
MAX_CONTEXT = 32768  # model's max_position_embeddings

GPU_TOTAL = 4 * 1024**3        # 4 GB card
WEIGHTS_SIZE = 1 * 1024**3     # ~1GB for Qwen2.5-0.5B in bfloat16 (measured earlier)
OVERHEAD = 200 * 1024**2       # reserve ~200MB for activations, fragmentation, CUDA context
CACHE_BUDGET = GPU_TOTAL - WEIGHTS_SIZE - OVERHEAD

print(f"Bytes per token (confirmed formula): {BYTES_PER_TOKEN:,}")
print(f"Total GPU memory: {GPU_TOTAL / 1e9:.2f} GB")
print(f"Reserved for weights: {WEIGHTS_SIZE / 1e9:.2f} GB")
print(f"Reserved for overhead: {OVERHEAD / 1e6:.0f} MB")
print(f"Remaining budget for KV cache: {CACHE_BUDGET / 1e9:.2f} GB\n")

# ---- Approach A: naive, worst-case reservation per request ----
naive_bytes_per_request = BYTES_PER_TOKEN * MAX_CONTEXT
naive_max_concurrent = CACHE_BUDGET // naive_bytes_per_request

print("=== Naive (contiguous, reserves full max-context per request) ===")
print(f"Memory reserved per request: {naive_bytes_per_request / 1e6:.1f} MB (for up to {MAX_CONTEXT:,} tokens)")
print(f"Max concurrent requests in budget: {naive_max_concurrent}\n")

# ---- Approach B: paged, realistic average conversation length ----
PAGE_SIZE = 16
AVG_CONVERSATION_LENGTH = 300  # a realistic short chat exchange, tokens

total_pages_available = CACHE_BUDGET // (PAGE_SIZE * BYTES_PER_TOKEN)
pages_per_avg_request = (AVG_CONVERSATION_LENGTH + PAGE_SIZE - 1) // PAGE_SIZE
paged_max_concurrent = total_pages_available // pages_per_avg_request

print(f"=== Paged (page_size={PAGE_SIZE}, average conversation length={AVG_CONVERSATION_LENGTH} tokens) ===")
print(f"Total pages available in budget: {total_pages_available:,}")
print(f"Pages needed per average request: {pages_per_avg_request} ({pages_per_avg_request * PAGE_SIZE} token capacity)")
print(f"Max concurrent requests in budget: {paged_max_concurrent}\n")

print(f"Paging allows ~{paged_max_concurrent / naive_max_concurrent:.0f}x more concurrent requests "
      f"in the same memory, when actual usage is far below the worst case.\n")

# ---- Live simulation: realistic mixed workload, actually using the allocator ----
print("=== Live simulation: mixed realistic conversation lengths ===")
import random
random.seed(42)

allocator = PageAllocator(num_pages=total_pages_available, page_size=PAGE_SIZE)
lengths = [random.randint(20, 800) for _ in range(200)]  # 200 simulated conversations, varied length

served = 0
total_pages_used = 0
for i, length in enumerate(lengths):
    seq_id = f"req-{i}"
    try:
        allocator.allocate_sequence(seq_id)
        newly = allocator.ensure_capacity(seq_id, num_tokens=length)
        total_pages_used += len(newly)
        served += 1
    except Exception as e:
        print(f"Stopped after {served} concurrent requests: {e}")
        break

if served == len(lengths):
    print(f"All {served} simulated requests (lengths 20-800 tokens, avg {sum(lengths)/len(lengths):.0f}) "
          f"fit simultaneously using {total_pages_used:,} of {total_pages_available:,} pages "
          f"({total_pages_used / total_pages_available * 100:.1f}% of budget).")
    naive_equivalent = served * (naive_bytes_per_request / (PAGE_SIZE * BYTES_PER_TOKEN))
    print(f"The naive approach could only have served {int(total_pages_available / (MAX_CONTEXT // PAGE_SIZE))} "
          f"of these same {served} requests in the same memory.")
