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
        self.num_groups = num_heads // num_kv_heads  # 14 // 2 = 7

        # These learn to produce Query, Key, Value from the input.
        # Qwen2 includes a bias term here, unlike many other models.
        self.q_proj = nn.Linear(hidden_size, num_heads * head_dim, bias=True)
        self.k_proj = nn.Linear(hidden_size, num_kv_heads * head_dim, bias=True)
        self.v_proj = nn.Linear(hidden_size, num_kv_heads * head_dim, bias=True)
        self.o_proj = nn.Linear(num_heads * head_dim, hidden_size, bias=False)

    def forward(self, x, cos, sin):
        batch, seq_len, _ = x.shape

        # Step 1: produce Query, Key, Value for every word
        q = self.q_proj(x).view(batch, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(batch, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(batch, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)

        # Step 2: rotate Query and Key using RoPE, so position info is baked in
        q = apply_rope(q, cos, sin)
        k = apply_rope(k, cos, sin)

        # Step 3: expand K and V so each of the 14 heads has a matching set
        # (groups of 7 heads share the same 2 KV sets)
        k = k.repeat_interleave(self.num_groups, dim=1)
        v = v.repeat_interleave(self.num_groups, dim=1)

        # Step 4: compute match scores between every Query and every Key
        scores = (q @ k.transpose(-2, -1)) / (self.head_dim ** 0.5)

        # Step 5: block attention to future words (causal mask)
        causal_mask = torch.triu(torch.ones(seq_len, seq_len, device=x.device), diagonal=1).bool()
        scores = scores.masked_fill(causal_mask, float("-inf"))

        # Step 6: turn scores into weights that sum to 1
        weights = F.softmax(scores, dim=-1)

        # Step 7: blend the Values according to those weights
        out = weights @ v

        # Step 8: recombine all 14 heads back into one 896-number fingerprint
        out = out.transpose(1, 2).contiguous().view(batch, seq_len, -1)
        return self.o_proj(out)
