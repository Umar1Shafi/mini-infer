from transformers import AutoConfig

config = AutoConfig.from_pretrained("Qwen/Qwen2.5-0.5B")
print(config)
