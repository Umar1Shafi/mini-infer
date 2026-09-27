import torch
from transformers import AutoModelForCausalLM, AutoConfig
from model.attention import GroupedQueryAttention
from model.rope import build_rope_cache

config = AutoConfig.from_pretrained("Qwen/Qwen2.5-0.5B")
model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-0.5B").float()

head_dim = config.hidden_size // config.num_attention_heads
seq_len = 5

real_attn = model.model.layers[0].self_attn
model = model.float()  # convert whole model to float32 for precise comparison

our_attn = GroupedQueryAttention(
    hidden_size=config.hidden_size,
    num_heads=config.num_attention_heads,
    num_kv_heads=config.num_key_value_heads,
    head_dim=head_dim,
)

# Copy the real, pretrained weights into our version
our_attn.q_proj.weight.data = real_attn.q_proj.weight.data.clone()
our_attn.q_proj.bias.data = real_attn.q_proj.bias.data.clone()
our_attn.k_proj.weight.data = real_attn.k_proj.weight.data.clone()
our_attn.k_proj.bias.data = real_attn.k_proj.bias.data.clone()
our_attn.v_proj.weight.data = real_attn.v_proj.weight.data.clone()
our_attn.v_proj.bias.data = real_attn.v_proj.bias.data.clone()
our_attn.o_proj.weight.data = real_attn.o_proj.weight.data.clone()

x = torch.randn(1, seq_len, config.hidden_size)
cos, sin = build_rope_cache(head_dim, max_seq_len=seq_len, theta=config.rope_parameters["rope_theta"])

position_ids = torch.arange(seq_len).unsqueeze(0)
real_cos, real_sin = model.model.rotary_emb(x, position_ids)

with torch.no_grad():
    our_output = our_attn(x, cos, sin)
    real_output = real_attn(x, position_embeddings=(real_cos, real_sin), attention_mask=None)[0]

max_diff = (our_output - real_output).abs().max().item()
print("Max difference:", max_diff)
print("Match!" if max_diff < 1e-3 else "MISMATCH - something's wrong")
