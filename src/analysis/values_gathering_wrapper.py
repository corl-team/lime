from typing import Optional, Tuple

import torch
import torch.nn.functional as F
from einops import rearrange
from torch import nn
from transformers.cache_utils import Cache
from transformers.models.llama.modeling_llama import (
    LlamaConfig,
    LlamaSdpaAttention,
    apply_rotary_pos_emb,
    repeat_kv,
)

from config import LIMeConfig
from src.lm.lime import LIMeAttentionProjection


class LlamaSdpaAttentionValues(LlamaSdpaAttention):
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[
            Tuple[torch.Tensor, torch.Tensor]
        ] = None,  # will become mandatory in v4.46
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        if output_attentions:
            return super().forward(
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_value=past_key_value,
                output_attentions=output_attentions,
                use_cache=use_cache,
                cache_position=cache_position,
                position_embeddings=position_embeddings,
            )

        bsz, q_len, _ = hidden_states.size()

        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        query_states = query_states.view(
            bsz, q_len, self.num_heads, self.head_dim
        ).transpose(1, 2)
        key_states = key_states.view(
            bsz, q_len, self.num_key_value_heads, self.head_dim
        ).transpose(1, 2)
        value_states = value_states.view(
            bsz, q_len, self.num_key_value_heads, self.head_dim
        ).transpose(1, 2)

        if position_embeddings is None:
            cos, sin = self.rotary_emb(value_states, position_ids)
        else:
            cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(
            query_states, key_states, cos, sin
        )

        if past_key_value is not None:
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            key_states, value_states = past_key_value.update(
                key_states, value_states, self.layer_idx, cache_kwargs
            )

        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)

        causal_mask = attention_mask
        if attention_mask is not None:
            causal_mask = causal_mask[:, :, :, : key_states.shape[-2]]

        if query_states.device.type == "cuda" and causal_mask is not None:
            query_states = query_states.contiguous()
            key_states = key_states.contiguous()
            value_states = value_states.contiguous()

        is_causal = True if causal_mask is None and q_len > 1 else False

        attn_output = torch.nn.functional.scaled_dot_product_attention(
            query_states,
            key_states,
            value_states,
            attn_mask=causal_mask,
            dropout_p=self.attention_dropout if self.training else 0.0,
            is_causal=is_causal,
        )

        self.value_states = rearrange(value_states, "b h t d -> b t (h d)")

        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(bsz, q_len, -1)

        attn_output = self.o_proj(attn_output)

        return attn_output, None, past_key_value


class LIMeSdpaAttentionValues(LlamaSdpaAttention):
    def __init__(
        self, llama_config: LlamaConfig, lime_config: LIMeConfig, layer_idx: int
    ):
        super().__init__(config=llama_config, layer_idx=layer_idx)

        del self.q_proj, self.k_proj, self.v_proj
        self.lime_config = lime_config
        self.lime_heads = llama_config.num_attention_heads
        self.k_proj_lime = LIMeAttentionProjection(llama_config)
        self.v_proj_lime = LIMeAttentionProjection(llama_config)
        self.q_proj_default = nn.Linear(
            self.hidden_size,
            self.num_heads * self.head_dim,
            bias=llama_config.attention_bias,
        )

    def forward(
        self,
        hidden_states_cur: torch.Tensor,
        hidden_states_past: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[
            Tuple[torch.Tensor, torch.Tensor]
        ] = None,  # will become mandatory in v4.46
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        if output_attentions:
            raise NotImplementedError("Default self attention is not implemented")
        bsz, nhead, q_len, _ = hidden_states_past.size()

        key_states = self.k_proj_lime(hidden_states_past)
        value_states = self.v_proj_lime(hidden_states_past)
        query_states = (
            self.q_proj_default(hidden_states_cur)
            .view(bsz, q_len, self.num_heads, self.head_dim)
            .transpose(1, 2)
        )  # [b, t, dh * hd]

        if position_embeddings is None:
            cos, sin = self.rotary_emb(value_states, position_ids)
        else:
            cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(
            query_states, key_states, cos, sin
        )

        if past_key_value is not None:
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            key_states, value_states = past_key_value.update(
                key_states, value_states, self.layer_idx, cache_kwargs
            )

        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)

        causal_mask = attention_mask
        if attention_mask is not None:
            causal_mask = causal_mask[:, :, :, : key_states.shape[-2]]
        if query_states.device.type == "cuda" and causal_mask is not None:
            query_states = query_states.contiguous()
            key_states = key_states.contiguous()
            value_states = value_states.contiguous()

        is_causal = True if causal_mask is None and q_len > 1 else False

        attn_output = torch.nn.functional.scaled_dot_product_attention(
            query_states,
            key_states,
            value_states,
            attn_mask=causal_mask,
            dropout_p=self.attention_dropout if self.training else 0.0,
            is_causal=is_causal,
        )

        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(bsz, q_len, -1)

        attn_output = self.o_proj(attn_output)

        self.value_states = rearrange(value_states, "b h t d -> b t (h d)")

        return attn_output, None, past_key_value


def value_wrapper(model, model_type):
    config = model.model.config
    if model_type.startswith("lime"):
        lime_config = model.model.lime_config
        model.model.layers[0].self_attn = LlamaSdpaAttentionValues(config, 0)
        for i in range(1, config.num_hidden_layers):
            model.model.layers[i].self_attn = LIMeSdpaAttentionValues(
                config, lime_config, i
            )
    else:
        for i in range(config.num_hidden_layers):
            model.model.layers[i].self_attn = LlamaSdpaAttentionValues(config, i)

    return model
