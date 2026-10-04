"""
Benchmarks the multi-page Triton paged-attention decode kernel against
the existing PyTorch forward_paged implementation, on a sequence long
enough to span many pages.
"""

import torch
import time
from transformers import AutoConfig, AutoModelForCausalLM
from model.attention import GroupedQueryAttention
from model.rope import build_rope_cache, apply_rope
from model.page_allocator import PageAllocator
from model.paged_cache import PagedKVCache
from model.paged_attention_triton import paged_attention_decode_multi_page

config = AutoConfig.from_pretrained("Qwen/Qwen2.5-0.5B")
hf_model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-0.5B").float().to("cuda")
head_dim = config.hidden_size // config.num_attention_heads
num_heads = config.num_attention_heads
num_kv_heads = config.num_key_value_heads
real_attn = hf_model.model.layers[0].self_attn

attn = GroupedQueryAttention(
    hidden_size=config.hidden_size, num_heads=num_heads,
    num_kv_heads=num_kv_heads, head_dim=head_dim,
).to("cuda")
attn.q_proj.weight.data = real_attn.q_proj.weight.data.clone()
attn.q_proj.bias.data = real_attn.q_proj.bias.data.clone()
attn.k_proj.weight.data = real_attn.k_proj.weight.data.clone()
attn.k_proj.bias.data = real_attn.k_proj.bias.data.clone()
attn.v_proj.weight.data = real_attn.v_proj.weight.data.clone()
attn.v_proj.bias.data = real_attn.v_proj.bias.data.clone()
attn.o_proj.weight.data = real_attn.o_proj.weight.data.clone()
attn.eval()

PAGE_SIZE = 16
ROPE_THETA = config.rope_parameters["rope_theta"]
PRIOR_LEN = 2000  # long decode history: ~125 pages
NUM_CALLS = 200

torch.manual_seed(7)
cache = PagedKVCache(
    num_layers=1, num_pages=256, page_size=PAGE_SIZE,
    num_kv_heads=num_kv_heads, head_dim=head_dim,
    dtype=torch.float32, device="cuda",
)
allocator = PageAllocator(num_pages=256, page_size=PAGE_SIZE)
allocator.allocate_sequence("seq-0")
allocator.ensure_capacity("seq-0", num_tokens=PRIOR_LEN)
table = allocator.get_page_table("seq-0")

prior_k = torch.randn(num_kv_heads, PRIOR_LEN, head_dim, device="cuda")
prior_v = torch.randn(num_kv_heads, PRIOR_LEN, head_dim, device="cuda")
cache.write(layer_idx=0, page_table=table, start_pos=0, k=prior_k, v=prior_v)

new_token_input = torch.randn(1, 1, config.hidden_size, device="cuda")

allocator.ensure_capacity("seq-0", num_tokens=PRIOR_LEN + 1)
table = allocator.get_page_table("seq-0")
print(f"Pages spanned: {len(table)} (prior_len={PRIOR_LEN}, page_size={PAGE_SIZE})")

cos_full, sin_full = build_rope_cache(head_dim, max_seq_len=PRIOR_LEN + 1, theta=ROPE_THETA, device="cuda")
cos_step, sin_step = cos_full[PRIOR_LEN:PRIOR_LEN+1], sin_full[PRIOR_LEN:PRIOR_LEN+1]

page_table_tensor = torch.tensor(table, dtype=torch.int32, device="cuda")
MAX_PAGES = len(table)

# ---------- Warm up + benchmark: PyTorch forward_paged ----------
with torch.no_grad():
    for _ in range(10):
        _ = attn.forward_paged(new_token_input, cos_step, sin_step, cache, layer_idx=0, page_table=table, start_pos=PRIOR_LEN)
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(NUM_CALLS):
        _ = attn.forward_paged(new_token_input, cos_step, sin_step, cache, layer_idx=0, page_table=table, start_pos=PRIOR_LEN)
    torch.cuda.synchronize()
    pytorch_time = (time.perf_counter() - start) / NUM_CALLS

# ---------- Warm up + benchmark: Triton kernel ----------
with torch.no_grad():
    x = new_token_input
    q = attn.q_proj(x).view(1, 1, num_heads, head_dim).transpose(1, 2)
    q = apply_rope(q, cos_step, sin_step)
    q_flat = q[0, :, 0, :].contiguous()
    k_pool_layer = cache.k_pool[0]
    v_pool_layer = cache.v_pool[0]

    for _ in range(10):
        _ = paged_attention_decode_multi_page(
            q_flat, k_pool_layer, v_pool_layer, page_table_tensor,
            seq_len=PRIOR_LEN + 1, num_heads=num_heads, num_kv_heads=num_kv_heads,
            head_dim=head_dim, page_size=PAGE_SIZE, max_pages=MAX_PAGES,
        )
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(NUM_CALLS):
        _ = paged_attention_decode_multi_page(
            q_flat, k_pool_layer, v_pool_layer, page_table_tensor,
            seq_len=PRIOR_LEN + 1, num_heads=num_heads, num_kv_heads=num_kv_heads,
            head_dim=head_dim, page_size=PAGE_SIZE, max_pages=MAX_PAGES,
        )
    torch.cuda.synchronize()
    triton_time = (time.perf_counter() - start) / NUM_CALLS

print(f"\nPyTorch forward_paged: {pytorch_time*1000:.4f} ms/call")
print(f"Triton kernel:         {triton_time*1000:.4f} ms/call")
print(f"Speedup: {pytorch_time/triton_time:.2f}x")

