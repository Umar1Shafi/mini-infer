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
    hidden_size=config.hidden_size, num_heads=config.num_attention_heads,
    num_kv_heads=config.num_key_value_heads, head_dim=head_dim,
)
attn.q_proj.weight.data = real_attn.q_proj.weight.data.clone()
attn.q_proj.bias.data = real_attn.q_proj.bias.data.clone()
attn.k_proj.weight.data = real_attn.k_proj.weight.data.clone()
attn.k_proj.bias.data = real_attn.k_proj.bias.data.clone()
attn.v_proj.weight.data = real_attn.v_proj.weight.data.clone()
attn.v_proj.bias.data = real_attn.v_proj.bias.data.clone()
attn.o_proj.weight.data = real_attn.o_proj.weight.data.clone()
attn.eval()

PAGE_SIZE = 4
ROPE_THETA = config.rope_parameters["rope_theta"]
NUM_KV_HEADS = config.num_key_value_heads
BATCH_SIZE = 4
# Deliberately different, messy lengths, including ones that don't align to PAGE_SIZE,
# and one sequence (index 2) that has NO prior history at all (a brand new request).
START_POSITIONS = [3, 11, 0, 7]

torch.manual_seed(123)

# ---------- Generate ALL random data ONCE, up front, and store it ----------
# Each sequence's prior history (if any) and its one new token this step.
prior_k = {}
prior_v = {}
for i in range(BATCH_SIZE):
    sp = START_POSITIONS[i]
    if sp > 0:
        prior_k[i] = torch.randn(NUM_KV_HEADS, sp, head_dim)
        prior_v[i] = torch.randn(NUM_KV_HEADS, sp, head_dim)

new_token_input = torch.randn(BATCH_SIZE, 1, config.hidden_size)  # the new token each sequence feeds in this step


def build_fresh_scenario():
    """Creates a brand-new allocator + paged cache, and writes the SAME prior histories into it."""
    cache = PagedKVCache(
        num_layers=1, num_pages=64, page_size=PAGE_SIZE,
        num_kv_heads=NUM_KV_HEADS, head_dim=head_dim,
        dtype=torch.float32, device="cpu",
    )
    allocator = PageAllocator(num_pages=64, page_size=PAGE_SIZE)
    tables = []
    for i in range(BATCH_SIZE):
        seq_id = f"seq-{i}"
        allocator.allocate_sequence(seq_id)
        sp = START_POSITIONS[i]
        if sp > 0:
            allocator.ensure_capacity(seq_id, num_tokens=sp)
            table = allocator.get_page_table(seq_id)
            cache.write(layer_idx=0, page_table=table, start_pos=0, k=prior_k[i], v=prior_v[i])
        else:
            table = allocator.get_page_table(seq_id)
        tables.append(table)
    return cache, allocator, tables


# ---------- Path A: run each sequence individually through forward_paged ----------
cache_a, allocator_a, tables_a = build_fresh_scenario()
individual_outputs = []
with torch.no_grad():
    for i in range(BATCH_SIZE):
        sp = START_POSITIONS[i]
        allocator_a.ensure_capacity(f"seq-{i}", num_tokens=sp + 1)
        table = allocator_a.get_page_table(f"seq-{i}")
        cos_full, sin_full = build_rope_cache(head_dim, max_seq_len=sp + 1, theta=ROPE_THETA)
        cos_step, sin_step = cos_full[sp:sp+1], sin_full[sp:sp+1]
        x_i = new_token_input[i:i+1]  # (1, 1, hidden_size)
        out = attn.forward_paged(x_i, cos_step, sin_step, cache_a, layer_idx=0, page_table=table, start_pos=sp)
        individual_outputs.append(out[0])  # drop batch dim -> (1, hidden_size)

# ---------- Path B: run all sequences together through forward_paged_batch ----------
cache_b, allocator_b, tables_b = build_fresh_scenario()
with torch.no_grad():
    for i in range(BATCH_SIZE):
        allocator_b.ensure_capacity(f"seq-{i}", num_tokens=START_POSITIONS[i] + 1)
    tables_b = [allocator_b.get_page_table(f"seq-{i}") for i in range(BATCH_SIZE)]

    batched_out = attn.forward_paged_batch(
        new_token_input, cache_b, layer_idx=0,
        page_tables=tables_b, start_positions=START_POSITIONS, rope_theta=ROPE_THETA,
    )  # (BATCH_SIZE, 1, hidden_size)

# ---------- Compare, sequence by sequence ----------
all_match = True
for i in range(BATCH_SIZE):
    a = individual_outputs[i]           # (1, hidden_size)
    b = batched_out[i]                  # (1, hidden_size)
    diff = (a - b).abs().max().item()
    status = "OK" if diff < 1e-5 else "MISMATCH"
    if diff >= 1e-5:
        all_match = False
    print(f"sequence {i} (start_pos={START_POSITIONS[i]}): max diff = {diff:.2e}  [{status}]")

print("\nALL SEQUENCES MATCH" if all_match else "\nFAILURE: batched and individual paths diverged")
