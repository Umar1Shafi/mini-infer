import torch
from model.page_allocator import PageAllocator
from model.paged_cache import PagedKVCache

def make_cache(num_pages=8, page_size=4, num_kv_heads=2, head_dim=4):
    return PagedKVCache(
        num_layers=1, num_pages=num_pages, page_size=page_size,
        num_kv_heads=num_kv_heads, head_dim=head_dim,
        dtype=torch.float32, device="cpu",
    ), page_size, num_kv_heads, head_dim

def test_single_page_roundtrip():
    cache, page_size, heads, dim = make_cache()
    alloc = PageAllocator(num_pages=8, page_size=page_size)
    alloc.allocate_sequence("req-1")
    alloc.ensure_capacity("req-1", num_tokens=3)
    table = alloc.get_page_table("req-1")

    k = torch.randn(heads, 3, dim)
    v = torch.randn(heads, 3, dim)
    cache.write(layer_idx=0, page_table=table, start_pos=0, k=k, v=v)

    k_read, v_read = cache.read(layer_idx=0, page_table=table, num_tokens=3)
    assert torch.equal(k, k_read), "K mismatch on single-page roundtrip"
    assert torch.equal(v, v_read), "V mismatch on single-page roundtrip"
    print("test_single_page_roundtrip: PASS")

def test_cross_page_boundary_roundtrip():
    """page_size=4, write 10 tokens -> spans 3 pages. Must reconstruct in exact order."""
    cache, page_size, heads, dim = make_cache()
    alloc = PageAllocator(num_pages=8, page_size=page_size)
    alloc.allocate_sequence("req-1")
    alloc.ensure_capacity("req-1", num_tokens=10)
    table = alloc.get_page_table("req-1")
    assert len(table) == 3, "10 tokens at page_size=4 should need 3 pages"

    k = torch.randn(heads, 10, dim)
    v = torch.randn(heads, 10, dim)
    cache.write(layer_idx=0, page_table=table, start_pos=0, k=k, v=v)

    k_read, v_read = cache.read(layer_idx=0, page_table=table, num_tokens=10)
    assert torch.equal(k, k_read), "K mismatch across page boundary"
    assert torch.equal(v, v_read), "V mismatch across page boundary"
    print("test_cross_page_boundary_roundtrip: PASS")

def test_incremental_decode_writes():
    """Simulates real usage: prefill with a few tokens, then append one token at a time."""
    cache, page_size, heads, dim = make_cache()
    alloc = PageAllocator(num_pages=8, page_size=page_size)
    alloc.allocate_sequence("req-1")

    all_k = []
    all_v = []

    # Prefill: 3 tokens at once
    alloc.ensure_capacity("req-1", num_tokens=3)
    table = alloc.get_page_table("req-1")
    k0 = torch.randn(heads, 3, dim)
    v0 = torch.randn(heads, 3, dim)
    cache.write(layer_idx=0, page_table=table, start_pos=0, k=k0, v=v0)
    all_k.append(k0)
    all_v.append(v0)

    # Decode: 5 more tokens, one at a time (this is the real generation pattern)
    pos = 3
    for _ in range(5):
        alloc.ensure_capacity("req-1", num_tokens=pos + 1)
        table = alloc.get_page_table("req-1")  # may have grown
        k_new = torch.randn(heads, 1, dim)
        v_new = torch.randn(heads, 1, dim)
        cache.write(layer_idx=0, page_table=table, start_pos=pos, k=k_new, v=v_new)
        all_k.append(k_new)
        all_v.append(v_new)
        pos += 1

    expected_k = torch.cat(all_k, dim=1)  # (heads, 8, dim)
    expected_v = torch.cat(all_v, dim=1)

    k_read, v_read = cache.read(layer_idx=0, page_table=table, num_tokens=8)
    assert torch.equal(expected_k, k_read), "K mismatch after incremental decode writes"
    assert torch.equal(expected_v, v_read), "V mismatch after incremental decode writes"
    print("test_incremental_decode_writes: PASS")

def test_two_sequences_do_not_interfere():
    cache, page_size, heads, dim = make_cache(num_pages=8, page_size=4)
    alloc = PageAllocator(num_pages=8, page_size=page_size)

    alloc.allocate_sequence("req-A")
    alloc.allocate_sequence("req-B")
    alloc.ensure_capacity("req-A", num_tokens=6)
    alloc.ensure_capacity("req-B", num_tokens=6)
    table_a = alloc.get_page_table("req-A")
    table_b = alloc.get_page_table("req-B")

    k_a = torch.randn(heads, 6, dim)
    v_a = torch.randn(heads, 6, dim)
    k_b = torch.randn(heads, 6, dim)
    v_b = torch.randn(heads, 6, dim)

    cache.write(layer_idx=0, page_table=table_a, start_pos=0, k=k_a, v=v_a)
    cache.write(layer_idx=0, page_table=table_b, start_pos=0, k=k_b, v=v_b)

    k_a_read, v_a_read = cache.read(layer_idx=0, page_table=table_a, num_tokens=6)
    k_b_read, v_b_read = cache.read(layer_idx=0, page_table=table_b, num_tokens=6)

    assert torch.equal(k_a, k_a_read), "req-A's K was corrupted by req-B"
    assert torch.equal(v_a, v_a_read), "req-A's V was corrupted by req-B"
    assert torch.equal(k_b, k_b_read), "req-B's K was corrupted by req-A"
    assert torch.equal(v_b, v_b_read), "req-B's V was corrupted by req-A"
    print("test_two_sequences_do_not_interfere: PASS")

def test_multi_layer_isolation():
    """Different layers must not read/write each other's data."""
    cache, page_size, heads, dim = make_cache(num_pages=8, page_size=4)
    cache.num_layers = 2
    cache.k_pool = torch.zeros((2, 8, page_size, heads, dim))
    cache.v_pool = torch.zeros((2, 8, page_size, heads, dim))

    alloc = PageAllocator(num_pages=8, page_size=page_size)
    alloc.allocate_sequence("req-1")
    alloc.ensure_capacity("req-1", num_tokens=4)
    table = alloc.get_page_table("req-1")

    k_layer0 = torch.randn(heads, 4, dim)
    v_layer0 = torch.randn(heads, 4, dim)
    k_layer1 = torch.randn(heads, 4, dim)
    v_layer1 = torch.randn(heads, 4, dim)

    cache.write(layer_idx=0, page_table=table, start_pos=0, k=k_layer0, v=v_layer0)
    cache.write(layer_idx=1, page_table=table, start_pos=0, k=k_layer1, v=v_layer1)

    k0_read, v0_read = cache.read(layer_idx=0, page_table=table, num_tokens=4)
    k1_read, v1_read = cache.read(layer_idx=1, page_table=table, num_tokens=4)

    assert torch.equal(k_layer0, k0_read), "Layer 0 K corrupted by layer 1 write"
    assert torch.equal(k_layer1, k1_read), "Layer 1 K corrupted by layer 0 write"
    assert not torch.equal(k0_read, k1_read), "Different layers' data should differ (random tensors)"
    print("test_multi_layer_isolation: PASS")

if __name__ == "__main__":
    test_single_page_roundtrip()
    test_cross_page_boundary_roundtrip()
    test_incremental_decode_writes()
    test_two_sequences_do_not_interfere()
    test_multi_layer_isolation()
    print("\nAll paged cache storage tests passed.")
