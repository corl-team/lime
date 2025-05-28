from typing import Optional, Tuple

import torch
from einops import rearrange
from transformers.models.llama.modeling_llama import (
    ALL_ATTENTION_FUNCTIONS,
    LlamaAttention,
    apply_rotary_pos_emb,
    eager_attention_forward,
)

from src.lm.lime import LIMeAttention


class LlamaAttentionValues(LlamaAttention):

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: Tuple[torch.Tensor, torch.Tensor],
        attention_mask: Optional[torch.Tensor],
        past_key_value=None,
        cache_position=None,
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)

        query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        key_states = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(
            query_states, key_states, cos, sin
        )

        if past_key_value is not None:
            # sin and cos are specific to RoPE models; cache_position needed for the static cache
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            key_states, value_states = past_key_value.update(
                key_states, value_states, self.layer_idx, cache_kwargs
            )

        attention_interface = eager_attention_forward
        if self.config._attn_implementation != "eager":
            attention_interface = ALL_ATTENTION_FUNCTIONS[
                self.config._attn_implementation
            ]

        attn_output, attn_weights = attention_interface(
            self,
            query_states,
            key_states,
            value_states,
            attention_mask,
            dropout=0.0 if not self.training else self.attention_dropout,
            scaling=self.scaling,
            **kwargs,
        )
        self.value_states = rearrange(value_states, "b h t d -> b t (h d)")
        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        attn_output = self.o_proj(attn_output)
        return attn_output, attn_weights


class LIMeAttentionValues(LIMeAttention):
    def forward(
        self,
        hidden_states: torch.Tensor,
        kv_buffer: torch.Tensor,
        position_embeddings: Tuple[torch.Tensor, torch.Tensor],
        attention_mask: Optional[torch.Tensor],
        past_key_value=None,
        cache_position=None,
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        input_shape = hidden_states.shape[:-1]
        B, T, D = hidden_states.shape
        H, num_kv_heads, Hd = (
            self.config.num_attention_heads,
            self.config.num_key_value_heads,
            self.head_dim,
        )
        q_hidden_shape = (B, T, H, Hd)

        # calculating queries from hidden as usual
        query_states = self.q_proj(hidden_states).view(q_hidden_shape).transpose(1, 2)

        cur_kv_states = torch.mm(hidden_states.view(-1, D), self.kv_proj_weight.t())

        kv_buffer = torch.cat(
            [
                kv_buffer,
                rearrange(
                    cur_kv_states, "(bt) (kv h) -> h (bt kv)", h=num_kv_heads, kv=2 * Hd
                ),
            ],
            dim=0,
        )

        # routing KV-cache to each head from all previous layers and heads
        key_value_states = self.lime_router(kv_buffer)
        key_value_states = key_value_states.view(num_kv_heads, B, T, 2 * Hd).permute(
            1, 0, 2, 3
        )
        key_states, value_states = (
            key_value_states[..., :Hd],
            key_value_states[..., Hd:],
        )

        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(
            query_states, key_states, cos, sin
        )

        if past_key_value is not None:
            # sin and cos are specific to RoPE models; cache_position needed for the static cache
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            key_states, value_states = past_key_value.update(
                key_states, value_states, self.layer_idx, cache_kwargs
            )

        attention_interface = eager_attention_forward
        if self.config._attn_implementation != "eager":
            attention_interface = ALL_ATTENTION_FUNCTIONS[
                self.config._attn_implementation
            ]

        attn_output, attn_weights = attention_interface(
            self,
            query_states,
            key_states,
            value_states,
            attention_mask,
            dropout=0.0 if not self.training else self.attention_dropout,
            scaling=self.scaling,
            **kwargs,
        )

        self.value_states = rearrange(value_states, "b h t d -> b t (h d)")
        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        attn_output = self.o_proj(attn_output)
        return attn_output, attn_weights, kv_buffer


def value_wrapper(model, model_type):
    config = model.model.config
    for i in range(config.num_hidden_layers):
        if model_type == "llama":
            model.model.layers[i].self_attn = LlamaAttentionValues(config, i)
        elif model_type == "lime":
            lime_config = model.model.lime_config
            model.model.layers[i].self_attn = LIMeAttentionValues(
                config, lime_config, i
            )
        else:
            raise NotImplementedError()
    return model
