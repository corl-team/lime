import math
from typing import List, Optional, Tuple, Union

import torch
import torch.nn.functional as F
from torch import nn
from transformers.cache_utils import Cache, DynamicCache
from transformers.modeling_outputs import BaseModelOutputWithPast
from transformers.models.llama.modeling_llama import (
    LlamaConfig,
    LlamaDecoderLayer,
    LlamaForCausalLM,
    LlamaModel,
    LlamaRMSNorm,
    LlamaSdpaAttention,
    apply_rotary_pos_emb,
    repeat_kv,
)

from config import LIMeConfig


class StaticRouterTopK(nn.Module):
    def __init__(self, llama_config, lime_config: LIMeConfig, layer_idx: int):
        super().__init__()
        assert layer_idx >= 1, "lime attention is not applicable on the 0th layer."

        # Initialize static weights
        w = (
            torch.randn(llama_config.num_attention_heads, layer_idx + 1)
            * lime_config.init_coef
        )
        self.static_weights = nn.Parameter(w)
        self.top_p = lime_config.top_p
        self.descending = lime_config.descending

        top_p_weights, self.pruned_count = self.get_pruning_mask()
        self.register_buffer("top_p_weights", top_p_weights)

    def update_mask(self):
        top_p_weights, self.pruned_count = self.get_pruning_mask()
        self.top_p_weights = top_p_weights

    def get_pruning_mask(self):
        normalized_weights = F.softmax(self.static_weights, dim=-1)
        sorted_weights, sorted_indices = torch.sort(
            normalized_weights, descending=self.descending, dim=1
        )
        cumsum_probs = torch.cumsum(sorted_weights, dim=1)
        cutoff_mask = cumsum_probs > self.top_p
        shifted_cutoff_mask = torch.zeros_like(cutoff_mask)
        shifted_cutoff_mask[:, 1:] = cutoff_mask[:, :-1]
        shifted_cutoff_mask[:, 0] = False

        filtered_sorted_weights = torch.where(
            shifted_cutoff_mask, torch.zeros_like(sorted_weights), sorted_weights
        )

        top_p_weights = torch.zeros_like(normalized_weights)
        top_p_weights.scatter_(1, sorted_indices, filtered_sorted_weights)

        return top_p_weights, shifted_cutoff_mask.sum()

    def forward(self, stacked_hiddens):
        return torch.einsum(
            "rl,lbtd->brtd",
            self.top_p_weights / self.top_p_weights.sum(dim=-1, keepdim=True),
            stacked_hiddens,
        )

    def extra_repr(self) -> str:
        return f"heads={self.static_weights.shape[0]}, n_repr={self.static_weights.shape[1]}, top_p={self.top_p}"


class StaticRouter(nn.Module):
    def __init__(
        self, llama_config: LlamaConfig, lime_config: LIMeConfig, layer_idx: int
    ):
        super().__init__()
        assert layer_idx >= 1, "lime attention is not applicable on the 0th layer."

        w = (
            torch.randn(llama_config.num_attention_heads, layer_idx + 1)
            * lime_config.init_coef
        )
        self.static_weights = nn.Parameter(w)

    def forward(self, stacked_hiddens):
        normalized_weights = self.static_weights.softmax(1)
        return torch.einsum("rl,lbtd->brtd", normalized_weights, stacked_hiddens)

    def extra_repr(self) -> str:
        return f"heads={self.static_weights.shape[0]}, n_repr={self.static_weights.shape[1]}"


class Reshape(nn.Module):
    def __init__(self, *new_shape: int):
        super(Reshape, self).__init__()
        self.new_shape = new_shape

    def forward(self, x):
        return x.view(*x.shape[:-1], *self.new_shape)

    def extra_repr(self):
        return str(self.new_shape)


class DynamicRouter(nn.Module):
    def __init__(
        self, llama_config: LlamaConfig, lime_config: LIMeConfig, layer_idx: int
    ):
        super().__init__()
        assert layer_idx >= 1, "LIMe is not applicable on the 0th layer."

        self.dynamic_proj = nn.Sequential(
            nn.Linear(
                llama_config.hidden_size,
                (layer_idx + 1) * llama_config.num_attention_heads,
                bias=False,
            ),
            Reshape(llama_config.num_attention_heads, layer_idx + 1),
        )

    def forward(self, stacked_hiddens):
        return torch.einsum(
            "btrl,lbtd->brtd",
            self.dynamic_proj(stacked_hiddens[-1]).softmax(-1),  # b t d -> b t r l
            stacked_hiddens,
        )

    def extra_repr(self) -> str:
        return self.dynamic_proj.extra_repr()


class LIMeAttentionProjection(nn.Module):
    def __init__(self, llama_config: LlamaConfig):
        super().__init__()
        self.llama_config = llama_config
        self.lime_kv_proj = nn.Parameter(
            torch.empty(
                1,
                llama_config.num_attention_heads,
                llama_config.hidden_size,
                llama_config.head_dim,
            )
        )
        nn.init.kaiming_uniform_(self.lime_kv_proj, a=math.sqrt(5))

    def forward(self, x):
        # x shape is [b, h, t, d] or [b 1 t d]
        # weight shape is [1, h, d, hd]
        return x @ self.lime_kv_proj

    def extra_repr(self) -> str:
        return f"heads={self.llama_config.num_attention_heads}, hidden_size={self.llama_config.hidden_size}, head_dim={self.llama_config.head_dim}"


class LIMeSdpaAttention(LlamaSdpaAttention):
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
            # sin and cos are specific to RoPE models; cache_position needed for the static cache
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

        return attn_output, None, past_key_value


class LIMeLayer(LlamaDecoderLayer):
    def __init__(
        self, llama_config: LlamaConfig, lime_config: LIMeConfig, layer_idx: int
    ):
        super().__init__(llama_config, layer_idx)
        self.lime_config = lime_config

        if lime_config.dynamic:
            router_cls = DynamicRouter
        else:
            router_cls = (
                StaticRouterTopK if lime_config.top_p is not None else StaticRouter
            )

        self.attention_router = router_cls(llama_config, lime_config, layer_idx)

        self.self_attn = LIMeSdpaAttention(llama_config, lime_config, layer_idx)

        self.input_layernorm_lime = LlamaRMSNorm(
            llama_config.hidden_size, eps=llama_config.rms_norm_eps
        )

    def forward(
        self,
        hidden_states_cur: torch.Tensor,
        hidden_states_past: Tuple[torch.Tensor],
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[
            Tuple[torch.Tensor, torch.Tensor]
        ] = None,  # will become mandatory in v4.46
        **kwargs,
    ) -> Tuple[
        torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]
    ]:
        residual = hidden_states_cur

        hidden_states_cur = self.input_layernorm(hidden_states_cur)

        attn_input = self.attention_router(
            self.input_layernorm_lime(hidden_states_past)
        )

        # Self Attention
        hidden_states, self_attn_weights, present_key_value = self.self_attn(
            hidden_states_cur=hidden_states_cur,
            hidden_states_past=attn_input,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
            cache_position=cache_position,
            position_embeddings=position_embeddings,
            **kwargs,
        )
        hidden_states_cur = residual + hidden_states
        # Fully Connected
        residual = hidden_states_cur
        hidden_states_cur = self.post_attention_layernorm(hidden_states_cur)
        hidden_states_cur = self.mlp(hidden_states_cur)
        hidden_states_cur = residual + hidden_states_cur

        if output_attentions:
            outputs += (self_attn_weights,)

        if use_cache:
            outputs += (present_key_value,)

        return hidden_states_cur


class LIMeModel(LlamaModel):
    def __init__(self, llama_config: LlamaConfig, lime_config: LIMeConfig):
        super().__init__(llama_config)

        layers = []
        for i in range(llama_config.num_hidden_layers):
            if i < lime_config.start_lime:
                layers += [LlamaDecoderLayer(llama_config, i)]
            else:
                layers += [LIMeLayer(llama_config, lime_config, i)]
        self.layers = nn.ModuleList(layers)
        self.lime_config = lime_config
        self.llama_config = llama_config
        self.post_init()

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Union[Cache, List[torch.FloatTensor]]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
    ) -> Union[Tuple, BaseModelOutputWithPast]:
        output_attentions = (
            output_attentions
            if output_attentions is not None
            else self.config.output_attentions
        )
        output_hidden_states = (
            output_hidden_states
            if output_hidden_states is not None
            else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError(
                "You must specify exactly one of input_ids or inputs_embeds"
            )

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        # kept for BC (non `Cache` `past_key_values` inputs)
        return_legacy_cache = False
        if use_cache and not isinstance(past_key_values, Cache):
            return_legacy_cache = True
            if past_key_values is None:
                past_key_values = DynamicCache()
            else:
                past_key_values = DynamicCache.from_legacy_cache(past_key_values)
        if cache_position is None:
            past_seen_tokens = (
                past_key_values.get_seq_length() if past_key_values is not None else 0
            )
            cache_position = torch.arange(
                past_seen_tokens,
                past_seen_tokens + inputs_embeds.shape[1],
                device=inputs_embeds.device,
            )
        if position_ids is None:
            position_ids = cache_position.unsqueeze(0)

        causal_mask = self._update_causal_mask(
            attention_mask,
            inputs_embeds,
            cache_position,
            past_key_values,
            output_attentions,
        )
        hidden_states_cur = inputs_embeds
        b, t, d = hidden_states_cur.shape

        # create position embeddings to be shared across the decoder layers
        position_embeddings = self.rotary_emb(hidden_states_cur, position_ids)

        # decoder layers
        hidden_states_past = hidden_states_cur.unsqueeze(0)

        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        next_decoder_cache = None

        for i, decoder_layer in enumerate(self.layers):
            # hidden_states_past += (hidden_states_cur,)
            if i > 0:
                # in efficient implementation hidden_states_past can be obtained directly from the gradient graph without memory overhead
                hidden_states_past = torch.vstack(
                    (hidden_states_past, hidden_states_cur.unsqueeze(0))
                )

            if output_hidden_states:
                all_hidden_states += (hidden_states_cur,)

            if i < self.lime_config.start_lime:
                layer_outputs = decoder_layer(
                    hidden_states_cur,
                    attention_mask=causal_mask,
                    position_ids=position_ids,
                    past_key_value=past_key_values,
                    output_attentions=output_attentions,
                    use_cache=use_cache,
                    cache_position=cache_position,
                    position_embeddings=position_embeddings,
                )
                hidden_states_cur = layer_outputs[0]
            else:
                hidden_states_cur = decoder_layer(
                    hidden_states_cur,
                    hidden_states_past,
                    attention_mask=causal_mask,
                    position_ids=position_ids,
                    past_key_value=past_key_values,
                    output_attentions=output_attentions,
                    use_cache=use_cache,
                    cache_position=cache_position,
                    position_embeddings=position_embeddings,
                )

        hidden_states = self.norm(hidden_states_cur)

        # add hidden states from the last decoder layer
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        next_cache = next_decoder_cache if use_cache else None
        if return_legacy_cache:
            next_cache = next_cache.to_legacy_cache()

        if not return_dict:
            return tuple(
                v
                for v in [hidden_states, next_cache, all_hidden_states, all_self_attns]
                if v is not None
            )
        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=next_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
        )


class LIMeForCausalLM(LlamaForCausalLM):
    _tied_weights_keys = ["lm_head.weight"]

    def __init__(self, llama_config, lime_config):
        super().__init__(llama_config)
        self.model = LIMeModel(llama_config, lime_config)
        self.vocab_size = llama_config.vocab_size
        self.lm_head = nn.Linear(
            llama_config.hidden_size, llama_config.vocab_size, bias=False
        )
        self.post_init()
