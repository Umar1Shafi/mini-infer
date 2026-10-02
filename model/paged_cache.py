import torch

class PagedKVCache:
    def __init__(self, num_layers, num_pages, page_size, num_kv_heads, head_dim, dtype, device):
        self.num_layers = num_layers
        self.num_pages = num_pages
        self.page_size = page_size
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim

        # One giant pool per layer, shared by every sequence.
        # Shape: (num_layers, num_pages, page_size, num_kv_heads, head_dim)
        shape = (num_layers, num_pages, page_size, num_kv_heads, head_dim)
        self.k_pool = torch.zeros(shape, dtype=dtype, device=device)
        self.v_pool = torch.zeros(shape, dtype=dtype, device=device)

    def write(self, layer_idx, page_table, start_pos, k, v):
        """
        Store new Key/Value tokens into their correct physical pages.

        k, v: shape (num_kv_heads, new_len, head_dim) — the NEW tokens only.
        page_table: ordered list of physical page indices owned by this sequence
                    (must already have enough pages allocated to hold start_pos + new_len).
        start_pos: the position (0-indexed) of the first NEW token in the sequence.
        """
        new_len = k.shape[1]
        for i in range(new_len):
            position = start_pos + i
            page_in_seq = position // self.page_size
            offset = position % self.page_size
            physical_page = page_table[page_in_seq]

            self.k_pool[layer_idx, physical_page, offset, :, :] = k[:, i, :]
            self.v_pool[layer_idx, physical_page, offset, :, :] = v[:, i, :]

    def read(self, layer_idx, page_table, num_tokens):
        """
        Reconstruct the full K/V history for a sequence as ONE contiguous tensor,
        by gathering across its (possibly scattered) physical pages.

        Returns k, v of shape (num_kv_heads, num_tokens, head_dim).
        """
        k_chunks = []
        v_chunks = []
        remaining = num_tokens

        for physical_page in page_table:
            if remaining <= 0:
                break
            take = min(self.page_size, remaining)
            k_chunks.append(self.k_pool[layer_idx, physical_page, :take, :, :])
            v_chunks.append(self.v_pool[layer_idx, physical_page, :take, :, :])
            remaining -= take

        # Each chunk is (take, num_kv_heads, head_dim) -> concat along token dim (dim=0)
        k_full = torch.cat(k_chunks, dim=0)  # (num_tokens, num_kv_heads, head_dim)
        v_full = torch.cat(v_chunks, dim=0)

        # Rearrange to (num_kv_heads, num_tokens, head_dim) to match attention's expected layout
        k_full = k_full.transpose(0, 1)
        v_full = v_full.transpose(0, 1)
        return k_full, v_full
