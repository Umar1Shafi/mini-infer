class OutOfMemoryError(Exception):
    """Raised when there are no free pages left to allocate."""
    pass

class PageAllocator:
    def __init__(self, num_pages, page_size):
        self.num_pages = num_pages
        self.page_size = page_size
        self.free_pages = list(range(num_pages))  # all pages start free
        self.sequence_tables = {}  # seq_id -> ordered list of page indices it owns

    def allocate_sequence(self, seq_id):
        """Register a new sequence with no pages yet."""
        if seq_id in self.sequence_tables:
            raise ValueError(f"Sequence {seq_id} already exists")
        self.sequence_tables[seq_id] = []

    def num_free_pages(self):
        return len(self.free_pages)

    def ensure_capacity(self, seq_id, num_tokens):
        """
        Make sure seq_id has enough allocated pages to hold num_tokens total.
        Allocates new pages from the free pool only if needed.
        Returns the list of newly allocated page indices (for logging/testing).
        """
        if seq_id not in self.sequence_tables:
            raise ValueError(f"Sequence {seq_id} not found, call allocate_sequence first")

        pages_needed = (num_tokens + self.page_size - 1) // self.page_size  # ceiling division
        current_pages = self.sequence_tables[seq_id]
        pages_to_add = pages_needed - len(current_pages)

        newly_allocated = []
        for _ in range(pages_to_add):
            if not self.free_pages:
                raise OutOfMemoryError(
                    f"No free pages left (requested capacity for {num_tokens} tokens, "
                    f"{self.num_free_pages()} pages free)"
                )
            page = self.free_pages.pop()
            current_pages.append(page)
            newly_allocated.append(page)

        return newly_allocated

    def get_page_table(self, seq_id):
        """Return the ordered list of physical page indices owned by this sequence."""
        return self.sequence_tables[seq_id]

    def free_sequence(self, seq_id):
        """Release all pages owned by seq_id back to the free pool."""
        pages = self.sequence_tables.pop(seq_id)
        self.free_pages.extend(pages)

    def token_slot(self, position):
        """
        Given a token's position within its sequence (0-indexed),
        return (which page in the sequence's list, offset within that page).
        """
        page_index_in_sequence = position // self.page_size
        offset_in_page = position % self.page_size
        return page_index_in_sequence, offset_in_page
