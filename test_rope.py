import torch
from transformers import AutoModelForCausalLM, AutoConfig
from model.rope import build_rope_cache

config = AutoConfig.from_pretrained("Qwen/Qwen2.5-0.5B")
model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-0.5B")

head_dim = config.hidden_size // config.num_attention_heads  # 896 // 14 = 64
seq_len = 5

# Our version: build the angle table directly
cos, sin = build_rope_cache(head_dim, max_seq_len=seq_len, theta=config.rope_parameters["rope_theta"])

# Official version: call the model's own rotary embedding module
dummy_hidden = torch.randn(1, seq_len, config.hidden_size)
position_ids = torch.arange(seq_len).unsqueeze(0)
real_cos, real_sin = model.model.rotary_emb(dummy_hidden, position_ids)

print("Our cos shape:", cos.shape)
print("Official cos shape:", real_cos.shape)

# Official cos/sin repeat each frequency twice (to cover all 64 dims at once)
# so we compare against just the first half to match our 32-wide table
max_diff_cos = (cos - real_cos[0, :, :head_dim // 2]).abs().max().item()
max_diff_sin = (sin - real_sin[0, :, :head_dim // 2]).abs().max().item()

print("Max cos difference:", max_diff_cos)
print("Max sin difference:", max_diff_sin)
print("Match!" if max_diff_cos < 1e-5 and max_diff_sin < 1e-5 else "MISMATCH - something's wrong")
