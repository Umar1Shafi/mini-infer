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

    def forward(self, input_ids, kv_caches=None, start_pos=0):
        batch, seq_len = input_ids.shape

        x = self.embed_tokens(input_ids)

        # Build RoPE angles for the CORRECT positions.
        # If we're generating word 6 alone, its position is 5 (0-indexed), not 0.
        cos_full, sin_full = build_rope_cache(
            self.head_dim, max_seq_len=start_pos + seq_len,
            theta=self.config.rope_parameters["rope_theta"],
            device=input_ids.device,
        )
        cos = cos_full[start_pos:start_pos + seq_len]
        sin = sin_full[start_pos:start_pos + seq_len]

        if kv_caches is None:
            kv_caches = [None] * len(self.layers)

        new_caches = []
        for layer, layer_cache in zip(self.layers, kv_caches):
            x, updated_cache = layer(x, cos, sin, layer_cache)
            new_caches.append(updated_cache)

        x = self.norm(x)
        logits = x @ self.embed_tokens.weight.T
        return logits, new_caches

    def load_pretrained_weights(self, hf_model):
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

    def forward_paged(self, input_ids, paged_cache, page_table, start_pos):
        """
        Generate logits using the paged KV cache instead of a contiguous one.
        Assumes batch=1 (one sequence at a time; concurrency comes from giving
        different sequences different page_tables, handled by the caller).
        """
        batch, seq_len = input_ids.shape
        assert batch == 1, "forward_paged expects one sequence at a time"

        x = self.embed_tokens(input_ids)

        cos_full, sin_full = build_rope_cache(
            self.head_dim, max_seq_len=start_pos + seq_len,
            theta=self.config.rope_parameters["rope_theta"],
            device=input_ids.device,
        )
        cos = cos_full[start_pos:start_pos + seq_len]
        sin = sin_full[start_pos:start_pos + seq_len]

        for layer_idx, layer in enumerate(self.layers):
            x = layer.forward_paged(x, cos, sin, paged_cache, layer_idx, page_table, start_pos)

        x = self.norm(x)
        logits = x @ self.embed_tokens.weight.T
        return logits
