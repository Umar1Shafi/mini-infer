import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-0.5B").cuda()
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B")

inputs = tokenizer("The capital of France is", return_tensors="pt").to("cuda")
output = model.generate(**inputs, max_new_tokens=10)
print(tokenizer.decode(output[0]))
