import torch
import torch.nn as nn
from model.rmsnorm import RMSNorm
from model.attention import GroupedQueryAttention
from model.feedforward import SwiGLU

class TransformerLayer(nn.Module):
    def __init__(self, hidden_size, num_heads, num_kv_heads, head_dim, intermediate_size, eps):
        super().__init__()
        self.input_layernorm = RMSNorm(hidden_size, eps)
        self.self_attn = GroupedQueryAttention(hidden_size, num_heads, num_kv_heads, head_dim)
        self.post_attention_layernorm = RMSNorm(hidden_size, eps)
        self.mlp = SwiGLU(hidden_size, intermediate_size)

    def forward(self, x, cos, sin):
        # Attention block, with residual connection
        residual = x
        x = self.input_layernorm(x)
        x = self.self_attn(x, cos, sin)
        x = residual + x

        # Feed-forward block, with residual connection
        residual = x
        x = self.post_attention_layernorm(x)
        x = self.mlp(x)
        x = residual + x

        return x
