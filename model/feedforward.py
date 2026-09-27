import torch
import torch.nn as nn
import torch.nn.functional as F

class SwiGLU(nn.Module):
    def __init__(self, hidden_size, intermediate_size):
        super().__init__()
        # Two separate expansions, done in parallel
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        # One projection back down to normal size
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)

    def forward(self, x):
        gate = F.silu(self.gate_proj(x))   # the "dimmer switch" pathway
        content = self.up_proj(x)          # the "raw content" pathway
        combined = gate * content          # gate controls how much content passes
        return self.down_proj(combined)    # shrink back to 896
