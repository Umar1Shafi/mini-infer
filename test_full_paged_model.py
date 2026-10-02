import torch
from transformers import AutoConfig, AutoTokenizer, AutoModelForCausalLM
from model.full_model import MiniQwen
from model.page_allocator import PageAllocator
from model.paged_cache import PagedKVCache

config = AutoConfig.from_pretrained("Qwen/Qwen2.5-0.5B")
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B")
hf_model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-0.5B").float()

model = MiniQwen(config)
model.load_pretrained_weights(hf_model)
model.eval()

prompt = "The capital of France is"
input_ids = tokenizer(prompt, return_tensors="pt")["input_ids"]
NUM_NEW = 20

# ---------- Reference: existing contiguous KV cache (already proven correct) ----------
with torch.no_grad():
    logits, caches = model(input_ids)
    next_token = logits[0, -1].argmax().view(1, 1)
    seq_ref = torch.cat([input_ids, next_token], dim=1)
    pos = input_ids.shape[1]
    for _ in range(NUM_NEW - 1):
        logits, caches = model(next_token, kv_caches=caches, start_pos=pos)
        next_token = logits[0, -1].argmax().view(1, 1)
        seq_ref = torch.cat([seq_ref, next_token], dim=1)
        pos += 1

# ---------- Paged generation ----------
PAGE_SIZE = 16  # deliberately does NOT evenly divide the prompt length (5 tokens)
with torch.no_grad():
    paged_cache = PagedKVCache(
        num_layers=config.num_hidden_layers, num_pages=16, page_size=PAGE_SIZE,
        num_kv_heads=config.num_key_value_heads,
        head_dim=config.hidden_size // config.num_attention_heads,
        dtype=torch.float32, device="cpu",
    )
    allocator = PageAllocator(num_pages=16, page_size=PAGE_SIZE)
    allocator.allocate_sequence("seq-0")

    prompt_len = input_ids.shape[1]
    allocator.ensure_capacity("seq-0", num_tokens=prompt_len)
    table = allocator.get_page_table("seq-0")
    logits = model.forward_paged(input_ids, paged_cache, table, start_pos=0)
    next_token = logits[0, -1].argmax().view(1, 1)
    seq_paged = torch.cat([input_ids, next_token], dim=1)
    pos = prompt_len

    for _ in range(NUM_NEW - 1):
        allocator.ensure_capacity("seq-0", num_tokens=pos + 1)
        table = allocator.get_page_table("seq-0")  # may have grown
        logits = model.forward_paged(next_token, paged_cache, table, start_pos=pos)
        next_token = logits[0, -1].argmax().view(1, 1)
        seq_paged = torch.cat([seq_paged, next_token], dim=1)
        pos += 1

text_ref = tokenizer.decode(seq_ref[0])
text_paged = tokenizer.decode(seq_paged[0])

print("Reference (contiguous cache):", text_ref)
print("Paged cache:                 ", text_paged)
print("\nToken IDs identical:", torch.equal(seq_ref, seq_paged))
print("Pages allocated for this 25-token sequence:", len(table), f"({len(table) * PAGE_SIZE} token capacity, {25} used)")
