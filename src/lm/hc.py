from dataclasses import dataclass
from typing import List, Optional, Tuple, Union

import torch
from torch import nn
from torch.nn import functional as F
from transformers.cache_utils import Cache, DynamicCache, StaticCache
from transformers.modeling_flash_attention_utils import _flash_attention_forward
from transformers.modeling_outputs import (
    BaseModelOutputWithPast,
    CausalLMOutputWithPast,
)
from transformers.models.llama.modeling_llama import (
    LlamaDecoderLayer,
    LlamaForCausalLM,
    LlamaModel,
)


class HyperConnection(nn.Module):
    def __init__(self, dim, rate, layer_id, dynamic):
        super(HyperConnection, self).__init__()
        self.rate = rate
        self.layer_id = layer_id
        self.dynamic = dynamic
        self.static_beta = nn.Parameter(torch.ones((rate,)))
        init_alpha0 = torch.zeros((rate, 1))
        init_alpha0[layer_id % rate, 0] = 1.0
        self.static_alpha = nn.Parameter(
            torch.cat([init_alpha0, torch.eye((rate))], dim=1)
        )
        if self.dynamic:
            self.dynamic_alpha_fn = nn.Parameter(torch.zeros((dim, rate + 1)))
            self.dynamic_alpha_scale = nn.Parameter(torch.ones(1) * 0.01)
            self.dynamic_beta_fn = nn.Parameter(torch.zeros((dim,)))
            self.dynamic_beta_scale = nn.Parameter(torch.ones(1) * 0.01)
            self.layer_norm = nn.LayerNorm(dim, elementwise_affine=False, bias=False)

    def width_connection(self, h):
        # get alpha and beta
        if self.dynamic:
            norm_h = self.layer_norm(h)
        if self.dynamic:
            wc_weight = norm_h @ self.dynamic_alpha_fn
            wc_weight = F.tanh(wc_weight)
            dynamic_alpha = wc_weight * self.dynamic_alpha_scale
            alpha = dynamic_alpha + self.static_alpha[None, None, ...]
        else:
            alpha = self.static_alpha[None, None, ...]
        if self.dynamic:
            dc_weight = (norm_h @ self.dynamic_beta_fn.unsqueeze(-1)).squeeze(-1)
            dc_weight = F.tanh(dc_weight)
            dynamic_beta = dc_weight * self.dynamic_beta_scale
            beta = dynamic_beta + self.static_beta[None, None, ...]
        else:
            beta = self.static_beta[None, None, ...]
        # width connection
        mix_h = alpha.transpose(-1, -2) @ h
        return mix_h, beta

    def depth_connection(self, mix_h, h_o, beta):
        h = torch.einsum("blh,bln->blnh", h_o, beta) + mix_h[..., 1:, :]
        return h


class LlamaHCDecoderLayer(LlamaDecoderLayer):
    def __init__(self, model_config, hc_config, layer_idx):
        super().__init__(config=model_config, layer_idx=layer_idx)
        self.hc_config = hc_config
        # self.self_attn = LLAMA_ATTENTION_CLASSES[model_config._attn_implementation](
        #     config=model_config, layer_idx=layer_idx
        # )
        self.hc_attn = HyperConnection(
            dim=model_config.hidden_size,
            rate=hc_config.rate,
            layer_id=layer_idx,
            dynamic=hc_config.dynamic,
        )
        self.hc_mlp = HyperConnection(
            dim=model_config.hidden_size,
            rate=hc_config.rate,
            layer_id=layer_idx,
            dynamic=hc_config.dynamic,
        )
        del self.input_layernorm
        del self.post_attention_layernorm
        self.attn_norm = nn.LayerNorm(
            model_config.hidden_size, elementwise_affine=False, bias=False
        )
        self.mlp_norm = nn.LayerNorm(
            model_config.hidden_size, elementwise_affine=False, bias=False
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
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

        if len(hidden_states.shape) == 3:
            # h = [b, l, dim] -> [b, l, rate, dim]
            hidden_states = hidden_states.unsqueeze(-2).expand(
                -1, -1, self.hc_config.rate, -1
            )

        mix_h, beta = self.hc_attn.width_connection(hidden_states)
        h = self.attn_norm(mix_h[..., 0, :])

        # Self Attention
        h, self_attn_weights = self.self_attn(
            hidden_states=h,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
            cache_position=cache_position,
            position_embeddings=position_embeddings,
            **kwargs,
        )

        h = self.hc_attn.depth_connection(mix_h, h, beta)

        # Fully Connected
        mix_h, beta = self.hc_mlp.width_connection(h)
        h = self.mlp_norm(mix_h[..., 0, :])
        h = self.mlp(h)
        h = self.hc_mlp.depth_connection(mix_h, h, beta)

        outputs = (h,)

        if output_attentions:
            outputs += (self_attn_weights,)

        return outputs


@dataclass
class LlamaHCModelOutputWithPast(BaseModelOutputWithPast):
    last_hidden_state: torch.FloatTensor = None
    loop_outputs: torch.FloatTensor = None
    past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None
    hidden_states: Optional[Tuple[torch.FloatTensor, ...]] = None
    attentions: Optional[Tuple[torch.FloatTensor, ...]] = None
    encoder_hidden_state: Optional[torch.FloatTensor] = None


@dataclass
class LlamaHCCausalLMOutputWithPast(CausalLMOutputWithPast):
    loss: Optional[torch.FloatTensor] = None
    logits: Optional[torch.FloatTensor] = None
    all_logits: Optional[torch.FloatTensor] = None
    past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None
    hidden_states: Optional[Tuple[torch.FloatTensor, ...]] = None
    attentions: Optional[Tuple[torch.FloatTensor, ...]] = None


class LlamaHCModel(LlamaModel):
    def __init__(self, model_config, hc_config):
        super().__init__(config=model_config)

        del self.layers
        self.layers = nn.ModuleList(
            [
                LlamaHCDecoderLayer(
                    model_config=model_config, hc_config=hc_config, layer_idx=i
                )
                for i in range(model_config.num_hidden_layers)
            ]
        )
        self.model_config = model_config

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
        return_dict: Optional[bool] = True,
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

        if self.gradient_checkpointing and self.training and use_cache:
            use_cache = False

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
        hidden_states = inputs_embeds

        # create position embeddings to be shared across the decoder layers
        position_embeddings = self.rotary_emb(hidden_states, position_ids)

        # decoder layers
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        next_decoder_cache = None

        for j in range(self.model_config.num_hidden_layers):
            if output_hidden_states:
                all_hidden_states += (hidden_states,)
            layer_outputs = self.layers[j](
                hidden_states,
                attention_mask=causal_mask,
                position_ids=position_ids,
                past_key_value=past_key_values,
                output_attentions=output_attentions,
                use_cache=use_cache,
                cache_position=cache_position,
                position_embeddings=position_embeddings,
            )

            hidden_states = layer_outputs[0]

            if use_cache:
                next_decoder_cache = layer_outputs[2 if output_attentions else 1]

            if output_attentions:
                all_self_attns += (layer_outputs[1],)

        hidden_states = hidden_states.sum(-2)
        hidden_states = self.norm(hidden_states)
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


class LlamaHCForCausalLM(LlamaForCausalLM):
    def __init__(self, model_config, hc_config):
        super().__init__(config=model_config)
        self.model = LlamaHCModel(model_config, hc_config)
        self.lm_head = nn.Linear(
            model_config.hidden_size, model_config.vocab_size, bias=False
        )
        self.post_init()
