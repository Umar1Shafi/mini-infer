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
