import math
from typing import Callable, Optional, Tuple

import torch
from einops import rearrange
from torch import nn
from transformers.cache_utils import Cache, DynamicCache
from transformers.modeling_flash_attention_utils import FlashAttentionKwargs
from transformers.modeling_outputs import BaseModelOutputWithPast
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
from transformers.models.llama.configuration_llama import LlamaConfig
from transformers.models.llama.modeling_llama import (
    LlamaAttention,
    LlamaDecoderLayer,
    LlamaForCausalLM,
    LlamaModel,
    apply_rotary_pos_emb,
    eager_attention_forward,
)
from transformers.processing_utils import Unpack
from transformers.utils import logging

logger = logging.get_logger(__name__)

from config import LIMeConfig


class StaticRouter(nn.Module):
    def __init__(
        self, llama_config: LlamaConfig, lime_config: LIMeConfig, layer_idx: int
    ):
        super().__init__()
        self.lime_config = lime_config
        self.R = llama_config.num_key_value_heads
        self.L = layer_idx + 1
        self.fan_in = self.L * self.R

        with torch.no_grad():
            bound = math.sqrt(3 / self.fan_in)
            w = torch.zeros(self.R, self.fan_in).uniform_(-bound, bound)
            w[:, -self.R :] = torch.eye(self.R)
            self.static_weights = nn.Parameter(w)

    def forward(self, stacked_last_hiddens):
        # stacked_last_hiddens: (L H) (B T 2 Hd)
        return self.static_weights.mm(stacked_last_hiddens)

    def extra_repr(self) -> str:
        return f"n_repr={self.fan_in}, heads={self.R}"


class FirstLayerRouter(nn.Module):
    def __init__(self, num_heads: int):
        super().__init__()
        self.num_heads = num_heads

    def forward(self, stacked_last_hiddens):
        return stacked_last_hiddens[: self.num_heads]


class LIMeAttention(LlamaAttention):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(self, llama_config: LlamaConfig, lime_config, layer_idx: int):
        super().__init__(config=llama_config, layer_idx=layer_idx)
        del self.k_proj, self.v_proj
        assert llama_config.attention_bias == False

        kv_proj_in_features = llama_config.hidden_size
        kv_proj_out_features = 2 * self.head_dim * llama_config.num_key_value_heads

        self.init_kv_proj(kv_proj_in_features, kv_proj_out_features)

        if layer_idx > 0:
            self.lime_router = StaticRouter(llama_config, lime_config, layer_idx)
        else:
            self.lime_router = FirstLayerRouter(
                num_heads=llama_config.num_key_value_heads
            )

        self.layer_idx: int = layer_idx

    def init_kv_proj(self, kv_proj_in_features: int, kv_proj_out_features: int) -> None:
        self.kv_proj_weight = nn.Parameter(
            torch.empty((kv_proj_out_features, kv_proj_in_features))
        )
        nn.init.kaiming_uniform_(self.kv_proj_weight, a=math.sqrt(5))

    def forward(
        self,
        hidden_states: torch.Tensor,
        kv_buffer: torch.Tensor,
        position_embeddings: Tuple[torch.Tensor, torch.Tensor],
        attention_mask: Optional[torch.Tensor],
        past_key_value: Optional[Cache] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs: Unpack[FlashAttentionKwargs],
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
        query_states = (
            self.q_proj(hidden_states).view(q_hidden_shape).transpose(1, 2)
        )  # B H T Hd

        cur_kv_states = torch.mm(
            hidden_states.view(-1, D), self.kv_proj_weight.t()
        )  # (B T) (2 Hd H)

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
        )  # B H T (2 Hd) -> B H T Hd, B H T Hd

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

        attention_interface: Callable = eager_attention_forward
        if self.config._attn_implementation != "eager":
            if self.config._attn_implementation == "sdpa" and kwargs.get(
                "output_attentions", False
            ):
                logger.warning_once(
                    "`torch.nn.functional.scaled_dot_product_attention` does not support `output_attentions=True`. Falling back to "
                    'eager attention. This warning can be removed using the argument `attn_implementation="eager"` when loading the model.'
                )
            else:
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

        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        attn_output = self.o_proj(attn_output)
        return attn_output, attn_weights, kv_buffer


class LIMeLayer(LlamaDecoderLayer):
    def __init__(
        self, llama_config: LlamaConfig, lime_config: LIMeConfig, layer_idx: int
    ):
        super().__init__(llama_config, layer_idx)
        self.lime_config = lime_config
        self.self_attn = LIMeAttention(llama_config, lime_config, layer_idx)

    def forward(
        self,
        hidden_states: torch.Tensor,
        kv_buffer: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[
            Tuple[torch.Tensor, torch.Tensor]
        ] = None,  # necessary, but kept here for BC
        **kwargs: Unpack[FlashAttentionKwargs],
    ) -> Tuple[
        torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]
    ]:
        residual = hidden_states

        hidden_states = self.input_layernorm(hidden_states)

        # Self Attention
        hidden_states, self_attn_weights, kv_buffer = self.self_attn(
            hidden_states=hidden_states,
            kv_buffer=kv_buffer,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
            cache_position=cache_position,
            position_embeddings=position_embeddings,
            **kwargs,
        )
        hidden_states = residual + hidden_states

        # Fully Connected
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        outputs = (hidden_states,)
        if output_attentions:
            outputs += (self_attn_weights,)

        outputs += (kv_buffer,)
        return outputs


class LIMeModel(LlamaModel):
    def __init__(self, llama_config: LlamaConfig, lime_config: LIMeConfig):
        super().__init__(llama_config)

        self.layers = nn.ModuleList(
            LIMeLayer(llama_config, lime_config, i)
            for i in range(llama_config.num_hidden_layers)
        )
        self.lime_config: LIMeConfig = lime_config
        self.llama_config: LlamaConfig = llama_config
        self.post_init()

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **flash_attn_kwargs: Unpack[FlashAttentionKwargs],
    ) -> BaseModelOutputWithPast:
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

        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError(
                "You must specify exactly one of input_ids or inputs_embeds"
            )

        if self.gradient_checkpointing and self.training and use_cache:
            logger.warning_once(
                "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`."
            )
            use_cache = False

        # TODO (joao): remove this exception in v4.56 -- it exists for users that try to pass a legacy cache
        if not isinstance(past_key_values, (type(None), Cache)):
            raise ValueError(
                "The `past_key_values` should be either a `Cache` object or `None`."
            )

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        if use_cache and past_key_values is None:
            past_key_values = DynamicCache()

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

        hidden_states = inputs_embeds

        position_embeddings = self.rotary_emb(hidden_states, position_ids)

        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None

        L, B, T, H, HD = (
            self.llama_config.num_hidden_layers,
            *hidden_states.shape[:-1],
            self.llama_config.num_key_value_heads,
            self.llama_config.head_dim,
        )
        with torch.no_grad():
            kv_buffer = torch.zeros(
                0 * H,
                B * T * 2 * HD,
                device=hidden_states.device,
                dtype=hidden_states.dtype,
                requires_grad=False,
            )
            layer_outputs = (kv_buffer,)

        for decoder_layer in self.layers[: self.config.num_hidden_layers]:
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            layer_outputs = decoder_layer(
                hidden_states,
                layer_outputs[-1],
                attention_mask=causal_mask,
                position_ids=position_ids,
                past_key_value=past_key_values,
                output_attentions=output_attentions,
                use_cache=use_cache,
                cache_position=cache_position,
                position_embeddings=position_embeddings,
                **flash_attn_kwargs,
            )

            hidden_states = layer_outputs[0]

            if output_attentions:
                all_self_attns += (layer_outputs[1],)

        hidden_states = self.norm(hidden_states)

        # add hidden states from the last decoder layer
        if output_hidden_states:
            all_hidden_states += (hidden_states,)
        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values if use_cache else None,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
        )


class LIMeForCausalLM(LlamaForCausalLM):
    def __init__(self, llama_config, lime_config):
        super().__init__(llama_config)
        self.model = LIMeModel(llama_config, lime_config)
        self.vocab_size = llama_config.vocab_size
        self.lm_head = nn.Linear(
            llama_config.hidden_size, llama_config.vocab_size, bias=False
        )
        self.post_init()
