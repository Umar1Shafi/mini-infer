import torch, time
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
del hf_model
model = model.to("cuda", dtype=torch.bfloat16).eval()

PAGE_SIZE = 16
EOS_ID = tokenizer.eos_token_id

PROMPTS = [
    ("The capital of France is", 30),
    ("Water boils at a temperature of", 30),
    ("The sky is", 30),
    ("Two plus two equals", 30),
    ("The largest planet in the solar system is", 30),
    ("Photosynthesis is the process by which", 30),
    ("The speed of light is approximately", 30),
    ("In the year 1969, humans first", 30),
]


def run_sequential():
    torch.cuda.synchronize()
    start = time.time()
    total_tokens = 0
    with torch.no_grad():
        for prompt, max_new in PROMPTS:
            input_ids = tokenizer(prompt, return_tensors="pt")["input_ids"].to("cuda")
            logits, caches = model(input_ids)
            next_token = logits[0, -1].argmax().view(1, 1)
            pos = input_ids.shape[1]
            generated = 1
            for _ in range(max_new - 1):
                if next_token.item() == EOS_ID:
                    break
                logits, caches = model(next_token, kv_caches=caches, start_pos=pos)
                next_token = logits[0, -1].argmax().view(1, 1)
                pos += 1
                generated += 1
            total_tokens += generated
    torch.cuda.synchronize()
    return time.time() - start, total_tokens


def run_continuous_batched(max_batch_size):
    paged_cache = PagedKVCache(
        num_layers=config.num_hidden_layers, num_pages=512, page_size=PAGE_SIZE,
        num_kv_heads=config.num_key_value_heads,
        head_dim=config.hidden_size // config.num_attention_heads,
        dtype=torch.bfloat16, device="cuda",
    )
    allocator = PageAllocator(num_pages=512, page_size=PAGE_SIZE)
    scheduler = Scheduler(max_batch_size=max_batch_size)

    for i, (prompt, max_new) in enumerate(PROMPTS):
        input_ids = tokenizer(prompt, return_tensors="pt")["input_ids"][0].tolist()
        req = Request(request_id=i, prompt_token_ids=input_ids, max_new_tokens=max_new, eos_token_id=EOS_ID)
        scheduler.add_request(req)
        allocator.allocate_sequence(i)

    total_tokens = 0
    torch.cuda.synchronize()
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
                total_tokens += 1

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
                total_tokens += 1

            finished = scheduler.step_evict_finished()
            for req in finished:
                allocator.free_sequence(req.request_id)

    torch.cuda.synchronize()
    return time.time() - start, total_tokens


# Warm-up (first CUDA call is always slow due to setup costs, don't time it)
run_continuous_batched(max_batch_size=4)

print(f"Serving {len(PROMPTS)} requests, up to 30 new tokens each\n")

seq_time, seq_tokens = run_sequential()
print(f"Sequential (one request at a time):")
print(f"  Total time: {seq_time:.2f}s, total tokens: {seq_tokens}, throughput: {seq_tokens / seq_time:.1f} tokens/sec\n")

for batch_size in [2, 4, 8]:
    batch_time, batch_tokens = run_continuous_batched(max_batch_size=batch_size)
    speedup = seq_time / batch_time
    print(f"Continuous batching (max_batch_size={batch_size}):")
    print(f"  Total time: {batch_time:.2f}s, total tokens: {batch_tokens}, throughput: {batch_tokens / batch_time:.1f} tokens/sec")
    print(f"  Speedup vs sequential: {speedup:.2f}x\n")
