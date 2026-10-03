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
    xf = x.float()  # do the rotation math in high precision
    out = (xf * cos) + (rotate_half(xf) * sin)
    return out.to(x.dtype)  # then go back to the model's working format
def apply_rope_batched(x, cos, sin):
    """
    Like apply_rope, but cos/sin carry a DIFFERENT angle per batch item
    (each sequence in a decode batch is at its own position).
    x:   (batch, heads, seq_len, head_dim)
    cos, sin: (batch, seq_len, head_dim/2)  -- note: per-batch-item, unlike apply_rope
    """
    cos = torch.cat((cos, cos), dim=-1).unsqueeze(1)  # (batch, 1, seq_len, head_dim)
    sin = torch.cat((sin, sin), dim=-1).unsqueeze(1)
    xf = x.float()
    out = (xf * cos) + (rotate_half(xf) * sin)
    return out.to(x.dtype)
