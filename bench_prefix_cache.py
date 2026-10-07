"""
Measures the memory savings from prefix caching in a realistic scenario:
many requests that all share a long, fixed system prompt, each followed
by a different short user message.

Compares pages used WITH prefix caching (system prompt pages shared
across all requests) vs WITHOUT (every request pays for its own copy).
"""

from model.page_allocator import PageAllocator
from model.prefix_cache import PrefixCache
import random

PAGE_SIZE = 16
NUM_REQUESTS = 200
SYSTEM_PROMPT_LEN = 300   # a long, fixed system prompt shared by every request
random.seed(0)

SYSTEM_PROMPT = list(range(1, SYSTEM_PROMPT_LEN + 1))  # fixed token ids, identical every time

def random_user_message(min_len=20, max_len=120):
    length = random.randint(min_len, max_len)
    # Random token ids, guaranteed not to collide with the system prompt's range
    return [random.randint(10000, 20000) for _ in range(length)]

# ---------- WITHOUT prefix caching: every request pays for the full prompt ----------
total_pages_naive = 0
for i in range(NUM_REQUESTS):
    tokens = SYSTEM_PROMPT + random_user_message()
    pages_needed = (len(tokens) + PAGE_SIZE - 1) // PAGE_SIZE
    total_pages_naive += pages_needed

# ---------- WITH prefix caching: system prompt pages are shared ----------
allocator = PageAllocator(num_pages=100000, page_size=PAGE_SIZE)  # plenty of headroom for this measurement
prefix_cache = PrefixCache(page_size=PAGE_SIZE)

total_pages_with_sharing = 0
shared_pages_reused_count = 0

for i in range(NUM_REQUESTS):
    tokens = SYSTEM_PROMPT + random_user_message()
    seq_id = f"req-{i}"
    allocator.allocate_sequence(seq_id)

    matched_pages, matched_len = prefix_cache.lookup(tokens)
    for page in matched_pages:
        allocator.attach_shared_page(seq_id, page)
        shared_pages_reused_count += 1

    allocator.ensure_capacity(seq_id, num_tokens=len(tokens))
    table = allocator.get_page_table(seq_id)
    prefix_cache.register(tokens, table)

    total_pages_with_sharing += len(table)
    # Only the FIRST request's system-prompt pages are ever newly allocated;
    # pages for positions it already registered are matched and reused by
    # every later request.

unique_physical_pages_used = allocator.num_pages - allocator.num_free_pages()

print(f"Requests: {NUM_REQUESTS}, shared system prompt: {SYSTEM_PROMPT_LEN} tokens, page size: {PAGE_SIZE}")
print(f"\nWithout prefix caching: {total_pages_naive} total page-allocations across all requests")
print(f"With prefix caching:    {unique_physical_pages_used} unique physical pages actually in use")
print(f"Page-table entries written (including shared references): {total_pages_with_sharing}")
print(f"Shared-page reuse events: {shared_pages_reused_count} "
      f"(requests 2..{NUM_REQUESTS} all reusing request 1's system-prompt pages)")
print(f"\nMemory reduction: {total_pages_naive / unique_physical_pages_used:.2f}x fewer physical pages needed")
