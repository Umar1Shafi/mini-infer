"""
Integration test: PageAllocator + PrefixCache + PagedKVCache working
together, with real K/V tensors (no model involved yet - this proves the
storage/sharing mechanism itself, the same way test_paged_cache.py proved
paging before attention was layered on top).

Scenario:
  Sequence A: 8 tokens (2 full pages)
  Sequence B: same first 8 tokens + 4 new tokens (shares A's 2 pages,
              gets 1 new private page)

Checks:
  - B's lookup correctly finds A's 2 pages as reusable
  - B does NOT rewrite the shared pages - it only writes its 4 new tokens
  - Reading back B's full 12-token history returns A's original data for
    the first 8 tokens and B's own data for the last 4
  - The shared pages' ref count is 2 while both A and B are alive
  - Freeing A does NOT corrupt or free the pages B is still using
  - Only freeing B (the last owner) returns them to the free pool
"""

import torch
from model.page_allocator import PageAllocator
from model.paged_cache import PagedKVCache
from model.prefix_cache import PrefixCache

PAGE_SIZE = 4
NUM_KV_HEADS = 2
HEAD_DIM = 8

torch.manual_seed(0)
allocator = PageAllocator(num_pages=8, page_size=PAGE_SIZE)
cache = PagedKVCache(num_layers=1, num_pages=8, page_size=PAGE_SIZE,
                      num_kv_heads=NUM_KV_HEADS, head_dim=HEAD_DIM,
                      dtype=torch.float32, device="cpu")
prefix_cache = PrefixCache(page_size=PAGE_SIZE)

# ---------- Sequence A: 8 tokens, 2 full pages ----------
tokens_a = [11, 12, 13, 14, 15, 16, 17, 18]
allocator.allocate_sequence("A")
allocator.ensure_capacity("A", num_tokens=len(tokens_a))
table_a = allocator.get_page_table("A")
assert len(table_a) == 2

k_a = torch.randn(NUM_KV_HEADS, len(tokens_a), HEAD_DIM)
v_a = torch.randn(NUM_KV_HEADS, len(tokens_a), HEAD_DIM)
cache.write(layer_idx=0, page_table=table_a, start_pos=0, k=k_a, v=v_a)
prefix_cache.register(tokens_a, table_a)

# ---------- Sequence B: shares A's 8 tokens, adds 4 new ones ----------
tokens_b = tokens_a + [19, 20, 21, 22]

matched_pages, matched_len = prefix_cache.lookup(tokens_b)
assert matched_pages == table_a, f"expected to reuse A's pages, got {matched_pages}"
assert matched_len == 8
print(f"B's lookup matched {len(matched_pages)} shared pages, {matched_len} tokens")

allocator.allocate_sequence("B")
for page in matched_pages:
    allocator.attach_shared_page("B", page)
allocator.ensure_capacity("B", num_tokens=len(tokens_b))  # adds exactly 1 new private page
table_b = allocator.get_page_table("B")
assert table_b[:2] == table_a, "B's page table must start with A's shared pages"
assert len(table_b) == 3
new_private_page = table_b[2]
print(f"B's page table: {table_b} (shared: {table_b[:2]}, private: {new_private_page})")

# B only computes/writes its 4 NEW tokens - the shared 8 are already correct in the pool
k_b_new = torch.randn(NUM_KV_HEADS, 4, HEAD_DIM)
v_b_new = torch.randn(NUM_KV_HEADS, 4, HEAD_DIM)
cache.write(layer_idx=0, page_table=table_b, start_pos=8, k=k_b_new, v=v_b_new)

# ---------- Verify B reads back the right data: A's old + B's new ----------
# cache.read() already returns (num_kv_heads, num_tokens, head_dim) - same layout as k_a/v_a
k_b_full, v_b_full = cache.read(layer_idx=0, page_table=table_b, num_tokens=12)

diff_shared = (k_b_full[:, :8, :] - k_a).abs().max().item()
diff_new = (k_b_full[:, 8:, :] - k_b_new).abs().max().item()
assert diff_shared == 0.0, f"shared portion should be bit-identical to A's data, diff={diff_shared}"
assert diff_new == 0.0, f"B's own new tokens should be bit-identical, diff={diff_new}"
print(f"B's reconstructed history matches: shared diff={diff_shared}, new diff={diff_new}")
# ---------- Ref counts while both alive ----------
assert allocator.ref_counts[table_a[0]] == 2
assert allocator.ref_counts[table_a[1]] == 2
assert allocator.ref_counts[new_private_page] == 1
print("Ref counts correct while A and B are both alive (shared pages: 2, private: 1)")

# ---------- Free A: B must be UNAFFECTED ----------
allocator.free_sequence("A")
assert table_a[0] not in allocator.free_pages, "shared page freed too early, B still needs it!"
assert table_a[1] not in allocator.free_pages

k_b_full_after, v_b_full_after = cache.read(layer_idx=0, page_table=table_b, num_tokens=12)
diff_after_free = (k_b_full_after - k_b_full).abs().max().item()
assert diff_after_free == 0.0, "B's data changed after freeing A - data corruption!"
print("After freeing A: B's data is untouched, still bit-identical")

# ---------- Free B: NOW the shared pages should return ----------
allocator.free_sequence("B")
assert table_a[0] in allocator.free_pages
assert table_a[1] in allocator.free_pages
assert new_private_page in allocator.free_pages
print("After freeing B: all pages correctly returned to the free pool")

print("\nAll prefix cache integration checks passed.")
