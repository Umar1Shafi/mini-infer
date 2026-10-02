import torch
from transformers import AutoConfig, AutoTokenizer, AutoModelForCausalLM
from model.full_model import MiniQwen

config = AutoConfig.from_pretrained("Qwen/Qwen2.5-0.5B")
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B")
hf = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-0.5B").float()
base = MiniQwen(config)
base.load_pretrained_weights(hf)
del hf

prompt_ids = tokenizer("The capital of France is", return_tensors="pt")["input_ids"].to("cuda")
N = 50

def run(model):
    with torch.no_grad():
        # no cache
        seq = prompt_ids.clone()
        for _ in range(N):
            logits, _ = model(seq)
            seq = torch.cat([seq, logits[0, -1].argmax().view(1, 1)], dim=1)
        a = seq[0]
        # with cache
        logits, caches = model(prompt_ids)
        nxt = logits[0, -1].argmax().view(1, 1)
        seq = torch.cat([prompt_ids, nxt], dim=1)
        pos = prompt_ids.shape[1]
        for _ in range(N - 1):
            logits, caches = model(nxt, kv_caches=caches, start_pos=pos)
            nxt = logits[0, -1].argmax().view(1, 1)
            seq = torch.cat([seq, nxt], dim=1)
            pos += 1
        b = seq[0]
    diff = (a != b).nonzero()
    return "identical" if len(diff) == 0 else f"first difference at token index {diff[0].item()} (prompt is {prompt_ids.shape[1]} tokens)"

for name, dtype in [("float32", torch.float32), ("bfloat16", torch.bfloat16)]:
    m = base.to("cuda", dtype=dtype).eval()
    print(f"{name}: {run(m)}")
    torch.cuda.empty_cache()
