from model.prefix_cache import PrefixCache

def test_no_match_on_empty_cache():
    cache = PrefixCache(page_size=4)
    matched_pages, matched_len = cache.lookup([1, 2, 3, 4, 5, 6, 7, 8])
    assert matched_pages == []
    assert matched_len == 0
    print("test_no_match_on_empty_cache: PASS")


def test_exact_full_match():
    cache = PrefixCache(page_size=4)
    tokens = [1, 2, 3, 4, 5, 6, 7, 8]  # exactly 2 full pages
    cache.register(tokens, page_table=[10, 11])

    matched_pages, matched_len = cache.lookup(tokens)
    assert matched_pages == [10, 11]
    assert matched_len == 8
    print("test_exact_full_match: PASS")


def test_partial_page_never_registered_or_matched():
    cache = PrefixCache(page_size=4)
    tokens = [1, 2, 3, 4, 5, 6]  # 1 full page + 2 leftover tokens (not a full page)
    cache.register(tokens, page_table=[10, 11])  # page 11 only holds 2 real tokens

    # Only the first full page should have been registered
    matched_pages, matched_len = cache.lookup([1, 2, 3, 4])
    assert matched_pages == [10]
    assert matched_len == 4

    # A request matching the full 6 tokens should NOT get credit for the
    # partial second page, since it was never safe to register
    matched_pages, matched_len = cache.lookup(tokens)
    assert matched_pages == [10]
    assert matched_len == 4
    print("test_partial_page_never_registered_or_matched: PASS")


def test_divergence_stops_the_match():
    cache = PrefixCache(page_size=4)
    tokens_a = [1, 2, 3, 4, 5, 6, 7, 8]
    cache.register(tokens_a, page_table=[10, 11])

    # Same first page, different second page -> should match page 0 only
    tokens_b = [1, 2, 3, 4, 99, 98, 97, 96]
    matched_pages, matched_len = cache.lookup(tokens_b)
    assert matched_pages == [10]
    assert matched_len == 4
    print("test_divergence_stops_the_match: PASS")


def test_different_first_page_matches_nothing():
    cache = PrefixCache(page_size=4)
    cache.register([1, 2, 3, 4, 5, 6, 7, 8], page_table=[10, 11])

    matched_pages, matched_len = cache.lookup([9, 9, 9, 9, 5, 6, 7, 8])
    assert matched_pages == []
    assert matched_len == 0
    print("test_different_first_page_matches_nothing: PASS")


def test_longer_new_prompt_matches_shared_prefix_only():
    cache = PrefixCache(page_size=4)
    cache.register([1, 2, 3, 4, 5, 6, 7, 8], page_table=[10, 11])

    # A longer prompt that shares the same first 8 tokens, plus 4 new ones
    new_tokens = [1, 2, 3, 4, 5, 6, 7, 8, 20, 21, 22, 23]
    matched_pages, matched_len = cache.lookup(new_tokens)
    assert matched_pages == [10, 11]
    assert matched_len == 8
    print("test_longer_new_prompt_matches_shared_prefix_only: PASS")


if __name__ == "__main__":
    test_no_match_on_empty_cache()
    test_exact_full_match()
    test_partial_page_never_registered_or_matched()
    test_divergence_stops_the_match()
    test_different_first_page_matches_nothing()
    test_longer_new_prompt_matches_shared_prefix_only()
    print("\nAll prefix cache tests passed.")

