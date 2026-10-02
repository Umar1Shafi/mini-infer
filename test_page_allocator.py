from model.page_allocator import PageAllocator, OutOfMemoryError

def test_basic_allocation():
    alloc = PageAllocator(num_pages=4, page_size=16)
    alloc.allocate_sequence("req-1")
    assert alloc.num_free_pages() == 4

    newly = alloc.ensure_capacity("req-1", num_tokens=5)
    assert len(newly) == 1, "5 tokens with page_size=16 needs exactly 1 page"
    assert alloc.num_free_pages() == 3
    print("test_basic_allocation: PASS")

def test_ceiling_division_exact_boundary():
    alloc = PageAllocator(num_pages=4, page_size=16)
    alloc.allocate_sequence("req-1")

    newly = alloc.ensure_capacity("req-1", num_tokens=16)  # exactly one full page
    assert len(newly) == 1, f"16 tokens should need exactly 1 page, got {len(newly)}"

    newly = alloc.ensure_capacity("req-1", num_tokens=17)  # one token into a second page
    assert len(newly) == 1, f"17 tokens should trigger allocating exactly 1 more page, got {len(newly)}"
    assert len(alloc.get_page_table("req-1")) == 2
    print("test_ceiling_division_exact_boundary: PASS")

def test_incremental_growth_no_realloc():
    """Growing token-by-token should only allocate a NEW page when crossing a page boundary."""
    alloc = PageAllocator(num_pages=4, page_size=16)
    alloc.allocate_sequence("req-1")

    total_new_pages = 0
    for token_count in range(1, 33):  # grow one token at a time up to 32
        newly = alloc.ensure_capacity("req-1", num_tokens=token_count)
        total_new_pages += len(newly)

    assert total_new_pages == 2, f"32 tokens should allocate exactly 2 pages total, allocated {total_new_pages}"
    print("test_incremental_growth_no_realloc: PASS")

def test_out_of_memory():
    alloc = PageAllocator(num_pages=2, page_size=16)
    alloc.allocate_sequence("req-1")
    alloc.ensure_capacity("req-1", num_tokens=32)  # uses both pages

    alloc.allocate_sequence("req-2")
    try:
        alloc.ensure_capacity("req-2", num_tokens=1)
        assert False, "Expected OutOfMemoryError but none was raised"
    except OutOfMemoryError:
        print("test_out_of_memory: PASS")

def test_free_and_reuse():
    alloc = PageAllocator(num_pages=2, page_size=16)
    alloc.allocate_sequence("req-1")
    alloc.ensure_capacity("req-1", num_tokens=32)  # uses both pages
    assert alloc.num_free_pages() == 0

    alloc.free_sequence("req-1")
    assert alloc.num_free_pages() == 2, "Freed pages should return to the pool"

    alloc.allocate_sequence("req-2")
    alloc.ensure_capacity("req-2", num_tokens=16)  # should succeed, reusing freed pages
    assert alloc.num_free_pages() == 1
    print("test_free_and_reuse: PASS")

def test_multiple_concurrent_sequences():
    alloc = PageAllocator(num_pages=4, page_size=16)
    alloc.allocate_sequence("req-1")
    alloc.allocate_sequence("req-2")

    alloc.ensure_capacity("req-1", num_tokens=10)
    alloc.ensure_capacity("req-2", num_tokens=20)

    assert len(alloc.get_page_table("req-1")) == 1
    assert len(alloc.get_page_table("req-2")) == 2
    assert alloc.num_free_pages() == 1

    # Make sure the two sequences never share a physical page
    pages_1 = set(alloc.get_page_table("req-1"))
    pages_2 = set(alloc.get_page_table("req-2"))
    assert pages_1.isdisjoint(pages_2), "Sequences must never share physical pages"
    print("test_multiple_concurrent_sequences: PASS")

def test_token_slot_mapping():
    alloc = PageAllocator(num_pages=4, page_size=16)

    assert alloc.token_slot(0) == (0, 0)
    assert alloc.token_slot(15) == (0, 15)
    assert alloc.token_slot(16) == (1, 0)    # crosses into the second page
    assert alloc.token_slot(37) == (2, 5)    # matches the worked example above
    print("test_token_slot_mapping: PASS")

if __name__ == "__main__":
    test_basic_allocation()
    test_ceiling_division_exact_boundary()
    test_incremental_growth_no_realloc()
    test_out_of_memory()
    test_free_and_reuse()
    test_multiple_concurrent_sequences()
    test_token_slot_mapping()
    print("\nAll page allocator tests passed.")
