"""
Verifies the multi-page Triton paged-attention kernel (online softmax,
looping over pages) against the proven forward_paged PyTorch implementation.

Uses PRIOR_LEN=37 with PAGE_SIZE=16 -> 3 pages (16, 16, 5), so this
deliberately exercises the multi-page loop including a partially-filled
final page, not just the single-page case already proven.
"""
import torch
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
PRIOR_LEN = 37  # spans 3 pages (16, 16, 5) -- deliberately NOT a multiple of PAGE_SIZE

torch.manual_seed(7)
cache = PagedKVCache(
    num_layers=1, num_pages=8, page_size=PAGE_SIZE,
    num_kv_heads=num_kv_heads, head_dim=head_dim,
    dtype=torch.float32, device="cuda",
)
allocator = PageAllocator(num_pages=8, page_size=PAGE_SIZE)
allocator.allocate_sequence("seq-0")
allocator.ensure_capacity("seq-0", num_tokens=PRIOR_LEN)
table = allocator.get_page_table("seq-0")

prior_k = torch.randn(num_kv_heads, PRIOR_LEN, head_dim, device="cuda")
prior_v = torch.randn(num_kv_heads, PRIOR_LEN, head_dim, device="cuda")
cache.write(layer_idx=0, page_table=table, start_pos=0, k=prior_k, v=prior_v)

new_token_input = torch.randn(1, 1, config.hidden_size, device="cuda")

# ---------- Reference: existing proven forward_paged ----------
with torch.no_grad():
    allocator.ensure_capacity("seq-0", num_tokens=PRIOR_LEN + 1)
    table = allocator.get_page_table("seq-0")
    print(f"Pages spanned: {len(table)} (prior_len={PRIOR_LEN}, page_size={PAGE_SIZE})")

    cos_full, sin_full = build_rope_cache(head_dim, max_seq_len=PRIOR_LEN + 1, theta=ROPE_THETA, device="cuda")
    cos_step, sin_step = cos_full[PRIOR_LEN:PRIOR_LEN+1], sin_full[PRIOR_LEN:PRIOR_LEN+1]
    reference_out = attn.forward_paged(new_token_input, cos_step, sin_step, cache, layer_idx=0, page_table=table, start_pos=PRIOR_LEN)

# ---------- Triton kernel path: compute q/k/v by hand (same proj + RoPE), then call the multi-page kernel ----------
with torch.no_grad():
    x = new_token_input
    q = attn.q_proj(x).view(1, 1, num_heads, head_dim).transpose(1, 2)
    k_new = attn.k_proj(x).view(1, 1, num_kv_heads, head_dim).transpose(1, 2)
    v_new = attn.v_proj(x).view(1, 1, num_kv_heads, head_dim).transpose(1, 2)
    q = apply_rope(q, cos_step, sin_step)
    k_new = apply_rope(k_new, cos_step, sin_step)

    # Cache already has PRIOR_LEN + 1 tokens written for layer 0 (forward_paged wrote this step above)
    k_pool_layer = cache.k_pool[0]  # (num_pages, page_size, num_kv_heads, head_dim)
    v_pool_layer = cache.v_pool[0]

    q_flat = q[0, :, 0, :].contiguous()  # (num_heads, head_dim)
    page_table_tensor = torch.tensor(table, dtype=torch.int32, device="cuda")
    MAX_PAGES = len(table)

    triton_out = paged_attention_decode_multi_page(
        q_flat, k_pool_layer, v_pool_layer, page_table_tensor,
        seq_len=PRIOR_LEN + 1, num_heads=num_heads, num_kv_heads=num_kv_heads,
        head_dim=head_dim, page_size=PAGE_SIZE, max_pages=MAX_PAGES,
    )
    triton_out = attn.o_proj(triton_out.view(1, -1))  # apply the same output projection

max_diff = (reference_out.view(-1) - triton_out.view(-1)).abs().max().item()
print(f"Max difference (multi-page Triton kernel vs. proven forward_paged): {max_diff:.2e}")
print("Match!" if max_diff < 1e-3 else "MISMATCH")
