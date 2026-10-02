import torch
from transformers import AutoConfig, AutoModelForCausalLM
from model.attention import GroupedQueryAttention
from model.rope import build_rope_cache
from model.page_allocator import PageAllocator
from model.paged_cache import PagedKVCache

config = AutoConfig.from_pretrained("Qwen/Qwen2.5-0.5B")
hf_model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-0.5B").float()
head_dim = config.hidden_size // config.num_attention_heads

real_attn = hf_model.model.layers[0].self_attn

attn = GroupedQueryAttention(
    hidden_size=config.hidden_size,
    num_heads=config.num_attention_heads,
    num_kv_heads=config.num_key_value_heads,
    head_dim=head_dim,
)
attn.q_proj.weight.data = real_attn.q_proj.weight.data.clone()
attn.q_proj.bias.data = real_attn.q_proj.bias.data.clone()
attn.k_proj.weight.data = real_attn.k_proj.weight.data.clone()
attn.k_proj.bias.data = real_attn.k_proj.bias.data.clone()
attn.v_proj.weight.data = real_attn.v_proj.weight.data.clone()
attn.v_proj.bias.data = real_attn.v_proj.bias.data.clone()
attn.o_proj.weight.data = real_attn.o_proj.weight.data.clone()
attn.eval()

SEQ_LEN = 5          # simulated prompt length
NUM_DECODE_STEPS = 10
PAGE_SIZE = 4        # deliberately small and NOT a multiple of SEQ_LEN, to force boundary crossings

torch.manual_seed(42)
full_input = torch.randn(1, SEQ_LEN + NUM_DECODE_STEPS, config.hidden_size)

cos, sin = build_rope_cache(head_dim, max_seq_len=SEQ_LEN + NUM_DECODE_STEPS, theta=config.rope_parameters["rope_theta"])

# ---------- Path A: existing contiguous cache (already proven correct) ----------
with torch.no_grad():
    prefill_input = full_input[:, :SEQ_LEN, :]
    cos_prefill, sin_prefill = cos[:SEQ_LEN], sin[:SEQ_LEN]
    out, cache = attn(prefill_input, cos_prefill, sin_prefill, kv_cache=None)
    contiguous_outputs = [out]

    for step in range(NUM_DECODE_STEPS):
        pos = SEQ_LEN + step
        token_input = full_input[:, pos:pos+1, :]
        cos_step, sin_step = cos[pos:pos+1], sin[pos:pos+1]
        out, cache = attn(token_input, cos_step, sin_step, kv_cache=cache)
        contiguous_outputs.append(out)

# ---------- Path B: paged cache ----------
with torch.no_grad():
    paged_cache = PagedKVCache(
        num_layers=1, num_pages=16, page_size=PAGE_SIZE,
        num_kv_heads=config.num_key_value_heads, head_dim=head_dim,
        dtype=torch.float32, device="cpu",
    )
    allocator = PageAllocator(num_pages=16, page_size=PAGE_SIZE)
    allocator.allocate_sequence("seq-0")

    prefill_input = full_input[:, :SEQ_LEN, :]
    allocator.ensure_capacity("seq-0", num_tokens=SEQ_LEN)
    table = allocator.get_page_table("seq-0")
    out = attn.forward_paged(prefill_input, cos_prefill, sin_prefill, paged_cache, layer_idx=0, page_table=table, start_pos=0)
    paged_outputs = [out]

    for step in range(NUM_DECODE_STEPS):
        pos = SEQ_LEN + step
        token_input = full_input[:, pos:pos+1, :]
        cos_step, sin_step = cos[pos:pos+1], sin[pos:pos+1]
        allocator.ensure_capacity("seq-0", num_tokens=pos + 1)
        table = allocator.get_page_table("seq-0")  # may have grown
        out = attn.forward_paged(token_input, cos_step, sin_step, paged_cache, layer_idx=0, page_table=table, start_pos=pos)
        paged_outputs.append(out)

# ---------- Compare every single step ----------
all_match = True
for i, (a, b) in enumerate(zip(contiguous_outputs, paged_outputs)):
    diff = (a - b).abs().max().item()
    label = "prefill" if i == 0 else f"decode step {i}"
    status = "OK" if diff < 1e-5 else "MISMATCH"
    if diff >= 1e-5:
        all_match = False
    print(f"{label}: max diff = {diff:.2e}  [{status}]")

print("\nALL STEPS MATCH" if all_match else "\nFAILURE: paged and contiguous caches diverged")
