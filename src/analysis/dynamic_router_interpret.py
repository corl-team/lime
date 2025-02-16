import torch
import torch.nn.functional as F
from torch import Tensor, nn
from tqdm import tqdm
from src.lm.lime import DynamicRouter, LIMeForCausalLM, LIMeLayer


def interpret_dynamic_router(
    embeddings: nn.Embedding,
    router: DynamicRouter,
    layer_norm,
    lm_head_norm,
    num_repr: int,
    k: int = 5,
) -> Tensor:
    unnormalized_embeds = embeddings.weight / lm_head_norm.weight
    # num_emb x D @ D x (H*L) -> num_emb, H, L
    logits = router.dynamic_proj(layer_norm(unnormalized_embeds)).view(
        embeddings.weight.shape[0], -1, num_repr
    )  # num_emb, H, L

    topk_indices = torch.topk(logits, k, dim=0, largest=True).indices  # [k, H, L]
    topk_indices = topk_indices.permute(1, 2, 0)  # [H, L, k]
    print(topk_indices.shape)
    return topk_indices


def interpret_dynamic_weights(
    model: LIMeForCausalLM, topk_nearest_embeds: int
) -> Tensor:
    nearest_embeds_for_layers = []
    for idx, layer in tqdm(enumerate(model.model.layers)):
        if isinstance(layer, LIMeLayer):
            topk_embeds = interpret_dynamic_router(
                model.model.embed_tokens,
                layer.attention_router,
                layer.input_layernorm_lime,
                model.model.norm,
                num_repr=idx + 1,
                k=topk_nearest_embeds,
            )  # [H, L, k]
            nearest_embeds_for_layers.append(topk_embeds)

    return nearest_embeds_for_layers


def get_most_activated_routes_desc(priors, topk: int = 200, start_from_layer: int = 4):
    tensors = []
    # layer 0 is embeds for notation
    assert start_from_layer >= 2, f"{start_from_layer} not lime layer"
    for layer_idx in range(start_from_layer - 2, len(priors)):
        router_weights = priors[layer_idx].detach().cpu().T  # L_to x heads
        router_weights[layer_idx + 1] = 0
        tensors.append(router_weights)
        shape = router_weights.shape

    padded = []
    layers, heads = shape
    for t in tensors:
        padded.append(F.pad(t, pad=(0, 0, 0, layers - t.shape[0])).unsqueeze(0))
    out = torch.vstack(padded)  # L_from x L_to x heads

    lower = 0
    multi_dim_indices_for_bins = []
    for _ in range(5):
        masked_probs = (out * (out >= max(lower, 1e-3)) * (out <= lower + 0.2)).view(-1)
        bin_size = (masked_probs > 0).sum()
        print(f"bin: [{lower:.1f}, {lower + 0.2:.1f}], {bin_size}")
        bin_topk = torch.unravel_index(
            masked_probs.topk(dim=-1, k=min(topk, bin_size)).indices, out.shape
        )
        multi_dim_indices_for_bins.append(
            (bin_topk[0] + start_from_layer, *bin_topk[1:])
        )
        lower += 0.2
    return multi_dim_indices_for_bins


def stack_all_router_hiddens(model: LIMeForCausalLM) -> Tensor:
    router_vectors = []
    keys = []
    for idx, layer in tqdm(enumerate(model.model.layers)):
        if isinstance(layer, LIMeLayer):
            vectors = (
                layer.attention_router.dynamic_proj[0].weight
                / layer.input_layernorm_lime.weight
            )  # (h l) d
            router_vectors.append(vectors)
            for l in range(idx + 1):
                for h in range(32):
                    keys += [f"{idx}_{l}_{h}"]

    return torch.vstack(router_vectors), keys  # B x d
