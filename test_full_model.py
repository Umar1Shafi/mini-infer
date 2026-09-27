import torch
from transformers import AutoModelForCausalLM, AutoConfig, AutoTokenizer
from model.full_model import MiniQwen

config = AutoConfig.from_pretrained("Qwen/Qwen2.5-0.5B")
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B")
hf_model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-0.5B").float()
hf_model.eval()

our_model = MiniQwen(config)
our_model.load_pretrained_weights(hf_model)
our_model.eval()

prompt = "The capital of France is"
inputs = tokenizer(prompt, return_tensors="pt")

with torch.no_grad():
    real_logits = hf_model(**inputs).logits
    our_logits = our_model(inputs["input_ids"])

max_diff = (real_logits - our_logits).abs().max().item()
print("Max logit difference:", max_diff)
print("Match!" if max_diff < 1e-2 else "MISMATCH - something's wrong")

# Also compare actual generated text, word by word, greedily
print("\n--- Generation test ---")
input_ids = inputs["input_ids"]
for _ in range(10):
    with torch.no_grad():
        logits = our_model(input_ids)
    next_token = logits[0, -1].argmax().unsqueeze(0).unsqueeze(0)
    input_ids = torch.cat([input_ids, next_token], dim=1)

print("Our model's output:", tokenizer.decode(input_ids[0]))
