import torch
from transformers import AutoModelForCausalLM
from model.rmsnorm import RMSNorm
from model.rmsnorm_triton import triton_rmsnorm

model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-0.5B").float().to("cuda")
real_norm = model.model.layers[0].input_layernorm

# Your existing, already-verified PyTorch RMSNorm (0.0 diff vs. official, back in Week 1)
our_norm = RMSNorm(hidden_size=real_norm.weight.shape[0], eps=real_norm.variance_epsilon).to("cuda")
our_norm.weight.data = real_norm.weight.data.clone()

# Test with a realistic shape: batch=2, seq_len=5, hidden_size=896
x = torch.randn(2, 5, real_norm.weight.shape[0], device="cuda")

with torch.no_grad():
    pytorch_output = our_norm(x)
    triton_output = triton_rmsnorm(x, our_norm.weight, eps=our_norm.eps)

max_diff = (pytorch_output - triton_output).abs().max().item()
print(f"Max difference (Triton vs. your proven PyTorch RMSNorm): {max_diff:.2e}")
print("Match!" if max_diff < 1e-5 else "MISMATCH")

# Also confirm it matches the OFFICIAL model's RMSNorm directly, for a complete chain of proof
with torch.no_grad():
    official_output = real_norm(x)
official_diff = (official_output - triton_output).abs().max().item()
print(f"Max difference (Triton vs. OFFICIAL Qwen2.5 RMSNorm): {official_diff:.2e}")
print("Match!" if official_diff < 1e-5 else "MISMATCH")
