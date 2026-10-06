class PrefixCache:
    """
    Tracks which already-written, FULLY FILLED pages correspond to a given
    exact prefix of token ids, so a new request sharing that prefix can
    reuse those pages' KV data instead of recomputing and storing it again.

    Only whole, filled pages are ever cached. A page that still has empty
    slots is still being written into by its owning sequence, so it isn't
    safe to hand out to anyone else yet.

    Matching is exact and page-aligned: two prompts share page i only if
    their first (i+1) * page_size tokens are identical. This is simpler
    than a true radix tree (no partial-page or branching matches) but is
    correct and is exactly where the actual memory/compute savings for
    something like a shared system prompt come from.
    """

    def __init__(self, page_size):
        self.page_size = page_size
        self.table = {}  # tuple(token_ids up to & including this page) -> physical_page

    def lookup(self, token_ids):
        """
        Given a new request's full prompt token ids, return:
          matched_pages: ordered list of physical pages that can be reused
          matched_len:   how many tokens (always a multiple of page_size)
                         those pages cover
        Stops at the first page that isn't an exact match (or isn't cached
        yet) - matching is a contiguous prefix, never a "skip and resume".
        """
        matched_pages = []
        pos = 0
        while pos + self.page_size <= len(token_ids):
            key = tuple(token_ids[:pos + self.page_size])
            if key in self.table:
                matched_pages.append(self.table[key])
                pos += self.page_size
            else:
                break
        return matched_pages, pos

    def register(self, token_ids, page_table):
        """
        After a sequence has written its tokens into page_table, record
        every FULLY FILLED page as available for future prefix matches.
        token_ids: the full list of token ids written so far for this sequence.
        page_table: this sequence's ordered list of physical pages.
        """
        pos = 0
        page_idx = 0
        while pos + self.page_size <= len(token_ids):
            key = tuple(token_ids[:pos + self.page_size])
            if key not in self.table:
                self.table[key] = page_table[page_idx]
            pos += self.page_size
            page_idx += 1
