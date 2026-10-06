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

def test_shared_page_ref_counting():
    """Two sequences sharing one physical page: the page must survive until
    BOTH release it, not just the first one."""
    allocator = PageAllocator(num_pages=4, page_size=16)

    allocator.allocate_sequence("seq-A")
    allocator.ensure_capacity("seq-A", num_tokens=16)  # allocates exactly 1 page
    shared_page = allocator.get_page_table("seq-A")[0]
    assert allocator.ref_counts[shared_page] == 1

    allocator.allocate_sequence("seq-B")
    allocator.attach_shared_page("seq-B", shared_page)
    assert allocator.ref_counts[shared_page] == 2, "ref count should be 2 after sharing"
    assert allocator.get_page_table("seq-B") == [shared_page]

    # Freeing seq-A must NOT return the page to the free pool - seq-B still needs it
    allocator.free_sequence("seq-A")
    assert allocator.ref_counts[shared_page] == 1
    assert shared_page not in allocator.free_pages, "page freed too early while still shared"

    # Now freeing seq-B (the last owner) SHOULD return it
    allocator.free_sequence("seq-B")
    assert allocator.ref_counts[shared_page] == 0
    assert shared_page in allocator.free_pages, "page should return to free pool once unreferenced"

    print("test_shared_page_ref_counting: PASS")


def test_three_way_sharing():
    """Three sequences sharing the same page: must take exactly three frees
    to return it, in any order."""
    allocator = PageAllocator(num_pages=4, page_size=16)

    allocator.allocate_sequence("seq-A")
    allocator.ensure_capacity("seq-A", num_tokens=16)
    shared_page = allocator.get_page_table("seq-A")[0]

    allocator.allocate_sequence("seq-B")
    allocator.attach_shared_page("seq-B", shared_page)
    allocator.allocate_sequence("seq-C")
    allocator.attach_shared_page("seq-C", shared_page)
    assert allocator.ref_counts[shared_page] == 3

    allocator.free_sequence("seq-B")
    assert shared_page not in allocator.free_pages
    allocator.free_sequence("seq-A")
    assert shared_page not in allocator.free_pages
    allocator.free_sequence("seq-C")
    assert shared_page in allocator.free_pages

    print("test_three_way_sharing: PASS")


def test_attach_shared_page_requires_registered_sequence():
    """attach_shared_page on an unregistered seq_id must fail loudly, the
    same way ensure_capacity already does."""
    allocator = PageAllocator(num_pages=4, page_size=16)
    try:
        allocator.attach_shared_page("ghost-seq", 0)
        assert False, "expected a ValueError for an unregistered sequence"
    except ValueError:
        pass

    print("test_attach_shared_page_requires_registered_sequence: PASS")


def test_private_pages_unaffected_by_sharing():
    """A sequence with a mix of shared + private pages must free correctly:
    the shared page waits for the other owner, the private page does not."""
    allocator = PageAllocator(num_pages=4, page_size=16)

    allocator.allocate_sequence("seq-A")
    allocator.ensure_capacity("seq-A", num_tokens=16)
    shared_page = allocator.get_page_table("seq-A")[0]

    allocator.allocate_sequence("seq-B")
    allocator.attach_shared_page("seq-B", shared_page)       # shared page
    allocator.ensure_capacity("seq-B", num_tokens=32)          # + 1 private page
    private_page = allocator.get_page_table("seq-B")[1]
    assert allocator.ref_counts[private_page] == 1

    allocator.free_sequence("seq-B")
    assert private_page in allocator.free_pages, "private page should free immediately"
    assert shared_page not in allocator.free_pages, "shared page still owned by seq-A"

    allocator.free_sequence("seq-A")
    assert shared_page in allocator.free_pages

    print("test_private_pages_unaffected_by_sharing: PASS")

if __name__ == "__main__":
    test_basic_allocation()
    test_ceiling_division_exact_boundary()
    test_incremental_growth_no_realloc()
    test_out_of_memory()
    test_free_and_reuse()
    test_multiple_concurrent_sequences()
    test_token_slot_mapping()
    test_shared_page_ref_counting()
    test_three_way_sharing()
    test_attach_shared_page_requires_registered_sequence()
    test_private_pages_unaffected_by_sharing()
    print("\nAll page allocator tests passed.")
