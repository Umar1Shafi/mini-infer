"""
Live demo: mini-infer serving 6 concurrent requests through the paged
cache + continuous-batching scheduler, printing each one's finished
text as soon as it completes - a visual, not just a benchmark number.
"""

import torch, time
from transformers import AutoConfig, AutoTokenizer, AutoModelForCausalLM
from model.full_model import MiniQwen
from model.page_allocator import PageAllocator
from model.paged_cache import PagedKVCache
from model.scheduler import Scheduler, Request, RequestStatus

MODEL_NAME = "Qwen/Qwen2.5-0.5B"
PAGE_SIZE = 16
MAX_BATCH_SIZE = 4

print("Loading mini-infer (from-scratch Qwen2.5-0.5B, paged cache, continuous batching)...")
config = AutoConfig.from_pretrained(MODEL_NAME)
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
hf_model = AutoModelForCausalLM.from_pretrained(MODEL_NAME).float()
model = MiniQwen(config)
model.load_pretrained_weights(hf_model)
del hf_model
model = model.to("cuda", dtype=torch.bfloat16).eval()

EOS_ID = tokenizer.eos_token_id
PROMPTS = [
    ("The capital of France is", 25),
    ("Water boils at a temperature of", 25),
    ("The three primary colors are", 25),
    ("A healthy breakfast usually includes", 25),
    ("The largest ocean on Earth is", 25),
    ("In the morning, most people like to", 25),
]

paged_cache = PagedKVCache(
    num_layers=config.num_hidden_layers, num_pages=512, page_size=PAGE_SIZE,
    num_kv_heads=config.num_key_value_heads,
    head_dim=config.hidden_size // config.num_attention_heads,
    dtype=torch.bfloat16, device="cuda",
)
allocator = PageAllocator(num_pages=512, page_size=PAGE_SIZE)
scheduler = Scheduler(max_batch_size=MAX_BATCH_SIZE)

for i, (prompt, max_new) in enumerate(PROMPTS):
    token_ids = tokenizer(prompt, return_tensors="pt")["input_ids"][0].tolist()
    req = Request(request_id=i, prompt_token_ids=token_ids, max_new_tokens=max_new, eos_token_id=EOS_ID)
    scheduler.add_request(req)
    allocator.allocate_sequence(i)

print(f"\nServing {len(PROMPTS)} prompts concurrently (max_batch_size={MAX_BATCH_SIZE})...\n")
print("-" * 70)

already_printed = set()
start = time.time()

with torch.no_grad():
    while scheduler.has_work():
        newly_admitted = scheduler.step_admit()
        for req in newly_admitted:
            prompt_ids = torch.tensor([req.prompt_token_ids], device="cuda")
            allocator.ensure_capacity(req.request_id, num_tokens=req.num_prompt_tokens)
            table = allocator.get_page_table(req.request_id)
            req.page_table = table
            logits = model.forward_paged(prompt_ids, paged_cache, table, start_pos=0)
            next_token = logits[0, -1].argmax().item()
            req.append_token(next_token)

        if not scheduler.running:
            continue

        running = scheduler.running
        input_ids_batch = torch.tensor([[req.generated_token_ids[-1]] for req in running], device="cuda")
        start_positions = [req.current_length() - 1 for req in running]
        for req in running:
            allocator.ensure_capacity(req.request_id, num_tokens=req.current_length() + 1)
            req.page_table = allocator.get_page_table(req.request_id)
        page_tables = [req.page_table for req in running]

        logits = model.forward_paged_batch(input_ids_batch, paged_cache, page_tables, start_positions)
        for i, req in enumerate(running):
            if req.status == RequestStatus.FINISHED:
                continue
            next_token = logits[i, -1].argmax().item()
            req.append_token(next_token)

        finished = scheduler.step_evict_finished()
        for req in finished:
            if req.request_id not in already_printed:
                text = tokenizer.decode(req.all_token_ids(), skip_special_tokens=True)
                elapsed = time.time() - start
                print(f"[{elapsed:5.2f}s] Request {req.request_id} finished:\n  \"{text}\"\n")
                already_printed.add(req.request_id)
            allocator.free_sequence(req.request_id)

total_time = time.time() - start
print("-" * 70)
print(f"All {len(PROMPTS)} requests completed in {total_time:.2f}s, served concurrently via continuous batching.")
