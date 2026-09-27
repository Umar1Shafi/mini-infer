import torch
import torch.nn as nn
from model.layer import TransformerLayer
from model.rmsnorm import RMSNorm
from model.rope import build_rope_cache

class MiniQwen(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        head_dim = config.hidden_size // config.num_attention_heads

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)

        self.layers = nn.ModuleList([
            TransformerLayer(
                hidden_size=config.hidden_size,
                num_heads=config.num_attention_heads,
                num_kv_heads=config.num_key_value_heads,
                head_dim=head_dim,
                intermediate_size=config.intermediate_size,
                eps=config.rms_norm_eps,
            )
            for _ in range(config.num_hidden_layers)
        ])

        self.norm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.head_dim = head_dim

    def forward(self, input_ids):
        batch, seq_len = input_ids.shape

        x = self.embed_tokens(input_ids)  # word IDs -> 896-number fingerprints

        cos, sin = build_rope_cache(
            self.head_dim, max_seq_len=seq_len,
            theta=self.config.rope_parameters["rope_theta"],
            device=input_ids.device,
        )

        for layer in self.layers:
            x = layer(x, cos, sin)

        x = self.norm(x)

        # Output head: reuse the embedding table (tied weights), transposed
        logits = x @ self.embed_tokens.weight.T
        return logits

    def load_pretrained_weights(self, hf_model):
        """Copy every real, trained weight from the official model into ours."""
        self.embed_tokens.weight.data = hf_model.model.embed_tokens.weight.data.clone()
        self.norm.weight.data = hf_model.model.norm.weight.data.clone()

        for i, layer in enumerate(self.layers):
            real_layer = hf_model.model.layers[i]

            layer.input_layernorm.weight.data = real_layer.input_layernorm.weight.data.clone()
            layer.post_attention_layernorm.weight.data = real_layer.post_attention_layernorm.weight.data.clone()

            layer.self_attn.q_proj.weight.data = real_layer.self_attn.q_proj.weight.data.clone()
            layer.self_attn.q_proj.bias.data = real_layer.self_attn.q_proj.bias.data.clone()
            layer.self_attn.k_proj.weight.data = real_layer.self_attn.k_proj.weight.data.clone()
            layer.self_attn.k_proj.bias.data = real_layer.self_attn.k_proj.bias.data.clone()
            layer.self_attn.v_proj.weight.data = real_layer.self_attn.v_proj.weight.data.clone()
            layer.self_attn.v_proj.bias.data = real_layer.self_attn.v_proj.bias.data.clone()
            layer.self_attn.o_proj.weight.data = real_layer.self_attn.o_proj.weight.data.clone()

            layer.mlp.gate_proj.weight.data = real_layer.mlp.gate_proj.weight.data.clone()
            layer.mlp.up_proj.weight.data = real_layer.mlp.up_proj.weight.data.clone()
            layer.mlp.down_proj.weight.data = real_layer.mlp.down_proj.weight.data.clone()
