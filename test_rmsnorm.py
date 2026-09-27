import torch
from transformers import AutoModelForCausalLM
from model.rmsnorm import RMSNorm

model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-0.5B")

# Grab the REAL RMSNorm from layer 0 of the official model
real_norm = model.model.layers[0].input_layernorm

# Build OUR version with the same size (896) and eps (1e-6)
our_norm = RMSNorm(hidden_size=896, eps=1e-6)

# Copy the real model's learned weight into ours, so we're comparing
# the same math, not different learned adjustments
our_norm.weight.data = real_norm.weight.data.clone()

# Feed both the same random "fake word" data: 1 sentence, 5 words, 896 numbers each
x = torch.randn(1, 5, 896)

with torch.no_grad():
    real_output = real_norm(x)
    our_output = our_norm(x)

max_diff = (real_output - our_output).abs().max().item()
print("Max difference:", max_diff)
print("Match!" if max_diff < 1e-5 else "MISMATCH - something's wrong")
