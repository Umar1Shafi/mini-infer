import torch
from transformers import AutoConfig, AutoTokenizer, AutoModelForCausalLM
from model.full_model import MiniQwen
from model.page_allocator import PageAllocator
from model.paged_cache import PagedKVCache
from model.scheduler import Scheduler, Request, RequestStatus

config = AutoConfig.from_pretrained("Qwen/Qwen2.5-0.5B")
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B")
hf_model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-0.5B").float()

model = MiniQwen(config)
model.load_pretrained_weights(hf_model)
model.eval()

PAGE_SIZE = 16
EOS_ID = tokenizer.eos_token_id

PROMPTS = [
    ("The capital of France is", 8),
    ("Water boils at a temperature of", 15),   # finishes later, tests slot-freeing
    ("The sky is", 5),                          # short, finishes FIRST, frees a slot
    ("Two plus two equals", 10),                # arrives mid-batch, like a late request
]

# ---------- Reference: run each prompt alone through the existing (proven) single-sequence path ----------
reference_outputs = {}
with torch.no_grad():
    for i, (prompt, max_new) in enumerate(PROMPTS):
        input_ids = tokenizer(prompt, return_tensors="pt")["input_ids"]
        logits, caches = model(input_ids)
        next_token = logits[0, -1].argmax().view(1, 1)
        seq = torch.cat([input_ids, next_token], dim=1)
        pos = input_ids.shape[1]
        for _ in range(max_new - 1):
            if next_token.item() == EOS_ID:
                break
            logits, caches = model(next_token, kv_caches=caches, start_pos=pos)
            next_token = logits[0, -1].argmax().view(1, 1)
            seq = torch.cat([seq, next_token], dim=1)
            pos += 1
        reference_outputs[i] = seq[0].tolist()

# ---------- Continuous batching: all 4 requests served together via the scheduler ----------
paged_cache = PagedKVCache(
    num_layers=config.num_hidden_layers, num_pages=256, page_size=PAGE_SIZE,
    num_kv_heads=config.num_key_value_heads,
    head_dim=config.hidden_size // config.num_attention_heads,
    dtype=torch.float32, device="cpu",
)
allocator = PageAllocator(num_pages=256, page_size=PAGE_SIZE)
scheduler = Scheduler(max_batch_size=3)  # deliberately SMALLER than 4 requests, forces slot-freeing to matter

requests_by_id = {}
for i, (prompt, max_new) in enumerate(PROMPTS):
    input_ids = tokenizer(prompt, return_tensors="pt")["input_ids"][0].tolist()
    req = Request(request_id=i, prompt_token_ids=input_ids, max_new_tokens=max_new, eos_token_id=EOS_ID)
    requests_by_id[i] = req
    scheduler.add_request(req)
    allocator.allocate_sequence(i)

batched_outputs = {}

with torch.no_grad():
    while scheduler.has_work():
        newly_admitted = scheduler.step_admit()

        # Handle prefill for any newly admitted request (one at a time, via the proven single-sequence path)
        for req in newly_admitted:
            prompt_ids = torch.tensor([req.prompt_token_ids])
            allocator.ensure_capacity(req.request_id, num_tokens=req.num_prompt_tokens)
            table = allocator.get_page_table(req.request_id)
            req.page_table = table
            logits = model.forward_paged(prompt_ids, paged_cache, table, start_pos=0)
            next_token = logits[0, -1].argmax().item()
            req.append_token(next_token)

        if not scheduler.running:
            continue

        # Batched decode step for every currently running request
        running = scheduler.running
        input_ids_batch = torch.tensor([[req.generated_token_ids[-1]] for req in running])
        start_positions = [req.current_length() - 1 for req in running]  # position of the LAST token already in cache

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
            batched_outputs[req.request_id] = req.all_token_ids()
            allocator.free_sequence(req.request_id)

# ---------- Compare ----------
all_match = True
for i, (prompt, _) in enumerate(PROMPTS):
    ref = reference_outputs[i]
    got = batched_outputs.get(i)
    match = (ref == got)
    if not match:
        all_match = False
    print(f"Request {i} ('{prompt}'): {'MATCH' if match else 'MISMATCH'}")
    if not match:
        print(f"  reference: {tokenizer.decode(ref)}")
        print(f"  batched:   {tokenizer.decode(got) if got else '(never finished)'}")

print("\nALL REQUESTS MATCH" if all_match else "\nFAILURE: continuous batching diverged from reference")
