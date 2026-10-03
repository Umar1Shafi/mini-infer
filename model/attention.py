from model.rope import apply_rope, apply_rope_batched
import torch
import torch.nn as nn
import torch.nn.functional as F
from model.rope import apply_rope

class GroupedQueryAttention(nn.Module):
    def __init__(self, hidden_size, num_heads, num_kv_heads, head_dim):
        super().__init__()
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.num_groups = num_heads // num_kv_heads

        self.q_proj = nn.Linear(hidden_size, num_heads * head_dim, bias=True)
        self.k_proj = nn.Linear(hidden_size, num_kv_heads * head_dim, bias=True)
        self.v_proj = nn.Linear(hidden_size, num_kv_heads * head_dim, bias=True)
        self.o_proj = nn.Linear(num_heads * head_dim, hidden_size, bias=False)

    def forward(self, x, cos, sin, kv_cache=None):
        batch, seq_len, _ = x.shape

        q = self.q_proj(x).view(batch, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(batch, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(batch, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)

        q = apply_rope(q, cos, sin)
        k = apply_rope(k, cos, sin)

        # --- NEW: use and update the cache ---
        if kv_cache is not None:
            past_k, past_v = kv_cache
            k = torch.cat([past_k, k], dim=2)
            v = torch.cat([past_v, v], dim=2)
        new_cache = (k, v)  # always save everything seen so far
        # --- end new ---

        k_expanded = k.repeat_interleave(self.num_groups, dim=1)
        v_expanded = v.repeat_interleave(self.num_groups, dim=1)

        scores = (q @ k_expanded.transpose(-2, -1)) / (self.head_dim ** 0.5)

        # Causal mask: only needed when processing multiple new words at once (prefill).
        # During decode (seq_len == 1), there's nothing "future" to hide from a single word.
        if seq_len > 1:
            kv_len = k_expanded.shape[2]
            causal_mask = torch.triu(
                torch.ones(seq_len, kv_len, device=x.device), diagonal=kv_len - seq_len + 1
            ).bool()
            scores = scores.masked_fill(causal_mask, float("-inf"))

        weights = F.softmax(scores, dim=-1)
        out = weights @ v_expanded

        out = out.transpose(1, 2).contiguous().view(batch, seq_len, -1)
        return self.o_proj(out), new_cache
 
    def forward_paged(self, x, cos, sin, paged_cache, layer_idx, page_table, start_pos):
        """
        Same attention computation as forward(), but reads/writes K/V through
        the paged cache instead of a single contiguous tensor.
        Assumes batch=1 (paging handles many concurrent sequences by giving
        each its own page_table, not by batching them into one tensor here).
        """
        batch, seq_len, _ = x.shape
        assert batch == 1, "forward_paged expects one sequence at a time (batch=1)"

        q = self.q_proj(x).view(batch, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(batch, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(batch, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)

        q = apply_rope(q, cos, sin)
        k = apply_rope(k, cos, sin)

        # Write only the NEW tokens' K/V into their physical pages
        k_new = k[0]  # drop batch dim -> (num_kv_heads, seq_len, head_dim)
        v_new = v[0]
        paged_cache.write(layer_idx, page_table, start_pos, k_new, v_new)

        # Read back the FULL history (old + new) by gathering across pages
        total_len = start_pos + seq_len
        k_full, v_full = paged_cache.read(layer_idx, page_table, total_len)
        k_full = k_full.unsqueeze(0)  # restore batch dim -> (1, num_kv_heads, total_len, head_dim)
        v_full = v_full.unsqueeze(0)

        k_expanded = k_full.repeat_interleave(self.num_groups, dim=1)
        v_expanded = v_full.repeat_interleave(self.num_groups, dim=1)

        scores = (q @ k_expanded.transpose(-2, -1)) / (self.head_dim ** 0.5)

        if seq_len > 1:
            kv_len = k_expanded.shape[2]
            causal_mask = torch.triu(
                torch.ones(seq_len, kv_len, device=x.device), diagonal=kv_len - seq_len + 1
            ).bool()
            scores = scores.masked_fill(causal_mask, float("-inf"))

        weights = F.softmax(scores, dim=-1)
        out = weights @ v_expanded

        out = out.transpose(1, 2).contiguous().view(batch, seq_len, -1)
        return self.o_proj(out)
    
    def forward_paged_batch(self, x_batch, paged_cache, layer_idx, page_tables, start_positions, rope_theta):
        """
        Batched DECODE step: one new token per sequence, sequences at different
        cache lengths. Pads shorter sequences' K/V to the batch max, masks the
        padding with -inf so it contributes nothing to softmax.

        x_batch: (batch_size, 1, hidden_size)
        page_tables: list of page tables, one per sequence, len == batch_size
        start_positions: list of ints (this new token's position per sequence)
        """
        from model.rope import build_rope_cache  # local import to avoid circularity concerns

        batch_size = x_batch.shape[0]
        assert x_batch.shape[1] == 1, "forward_paged_batch handles one new token per sequence (decode only)"
        assert len(page_tables) == batch_size == len(start_positions)

        q = self.q_proj(x_batch).view(batch_size, 1, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x_batch).view(batch_size, 1, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x_batch).view(batch_size, 1, self.num_kv_heads, self.head_dim).transpose(1, 2)

        # Per-sequence RoPE angles: each sequence is at its own position
        max_pos = max(start_positions) + 1
        cos_full, sin_full = build_rope_cache(self.head_dim, max_seq_len=max_pos, theta=rope_theta, device=x_batch.device)
        positions_tensor = torch.tensor(start_positions, device=x_batch.device)
        cos = cos_full[positions_tensor].unsqueeze(1)  # (batch, 1, head_dim/2)
        sin = sin_full[positions_tensor].unsqueeze(1)

        q = apply_rope_batched(q, cos, sin)
        k = apply_rope_batched(k, cos, sin)

        # Write this step's new token into each sequence's own pages, then
        # read back each sequence's FULL history (different lengths, hence the loop)
        k_list, v_list, lengths = [], [], []
        for i in range(batch_size):
            k_i = k[i]  # (num_kv_heads, 1, head_dim)
            v_i = v[i]
            paged_cache.write(layer_idx, page_tables[i], start_positions[i], k_i, v_i)
            total_len = start_positions[i] + 1
            k_full, v_full = paged_cache.read(layer_idx, page_tables[i], total_len)
            k_list.append(k_full)
            v_list.append(v_full)
            lengths.append(total_len)

        # Pad every sequence's history to the batch's longest, and mark which
        # positions are REAL vs. padding (padding gets masked to -inf before softmax)
        max_len = max(lengths)
        k_padded = torch.zeros(batch_size, self.num_kv_heads, max_len, self.head_dim, dtype=x_batch.dtype, device=x_batch.device)
        v_padded = torch.zeros_like(k_padded)
        valid_mask = torch.zeros(batch_size, max_len, dtype=torch.bool, device=x_batch.device)
        for i in range(batch_size):
            L = lengths[i]
            k_padded[i, :, :L, :] = k_list[i]
            v_padded[i, :, :L, :] = v_list[i]
            valid_mask[i, :L] = True

        k_expanded = k_padded.repeat_interleave(self.num_groups, dim=1)
        v_expanded = v_padded.repeat_interleave(self.num_groups, dim=1)

        scores = (q @ k_expanded.transpose(-2, -1)) / (self.head_dim ** 0.5)  # (batch, heads, 1, max_len)

        pad_mask = (~valid_mask).unsqueeze(1).unsqueeze(1)  # (batch, 1, 1, max_len)
        scores = scores.masked_fill(pad_mask, float("-inf"))

        weights = F.softmax(scores, dim=-1)
        out = weights @ v_expanded  # (batch, heads, 1, head_dim)

        out = out.transpose(1, 2).contiguous().view(batch_size, 1, -1)
        return self.o_proj(out)
