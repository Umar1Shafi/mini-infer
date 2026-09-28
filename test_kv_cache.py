import torch, time
from transformers import AutoConfig, AutoTokenizer
from model.full_model import MiniQwen
from transformers import AutoModelForCausalLM

config = AutoConfig.from_pretrained("Qwen/Qwen2.5-0.5B")
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B")
hf_model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-0.5B").float()

our_model = MiniQwen(config)
our_model.load_pretrained_weights(hf_model)
our_model.eval()

prompt = "The capital of France is"
inputs = tokenizer(prompt, return_tensors="pt")
input_ids = inputs["input_ids"]

NUM_NEW_TOKENS = 20

# ---- Method 1: NO cache (old way, reprocess everything each time) ----
start = time.time()
seq = input_ids.clone()
with torch.no_grad():
    for _ in range(NUM_NEW_TOKENS):
        logits, _ = our_model(seq)  # no cache passed, reprocesses everything
        next_token = logits[0, -1].argmax().unsqueeze(0).unsqueeze(0)
        seq = torch.cat([seq, next_token], dim=1)
no_cache_time = time.time() - start
no_cache_text = tokenizer.decode(seq[0])

# ---- Method 2: WITH cache (new way) ----
start = time.time()
with torch.no_grad():
    # Prefill: process the whole prompt once, build the initial cache
    logits, caches = our_model(input_ids, kv_caches=None, start_pos=0)
    next_token = logits[0, -1].argmax().unsqueeze(0).unsqueeze(0)
    seq = torch.cat([input_ids, next_token], dim=1)
    pos = input_ids.shape[1]

    # Decode: one new word at a time, reusing the cache
    for _ in range(NUM_NEW_TOKENS - 1):
        logits, caches = our_model(next_token, kv_caches=caches, start_pos=pos)
        next_token = logits[0, -1].argmax().unsqueeze(0).unsqueeze(0)
        seq = torch.cat([seq, next_token], dim=1)
        pos += 1
cache_time = time.time() - start
cache_text = tokenizer.decode(seq[0])

print("No-cache output: ", no_cache_text)
print("With-cache output:", cache_text)
print("Outputs match:", no_cache_text == cache_text)
print(f"\nNo-cache time:   {no_cache_time:.3f}s")
print(f"With-cache time: {cache_time:.3f}s")
print(f"Speedup: {no_cache_time / cache_time:.2f}x")
