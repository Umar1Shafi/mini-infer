import torch

def build_rope_cache(head_dim, max_seq_len, theta=1000000.0, device="cpu"):
    inv_freq = 1.0 / (theta ** (torch.arange(0, head_dim, 2, device=device).float() / head_dim))
    positions = torch.arange(max_seq_len, device=device).float()
    freqs = torch.outer(positions, inv_freq)
    cos = freqs.cos()
    sin = freqs.sin()
    return cos, sin

def rotate_half(x):
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)

def apply_rope(x, cos, sin):
    cos = torch.cat((cos, cos), dim=-1).unsqueeze(0).unsqueeze(0)
    sin = torch.cat((sin, sin), dim=-1).unsqueeze(0).unsqueeze(0)
    return (x * cos) + (rotate_half(x) * sin)
