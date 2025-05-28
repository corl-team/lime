from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from einops import rearrange
from sklearn.manifold import TSNE
from sklearn.preprocessing import normalize
from tqdm.auto import tqdm


def gather_same_hiddens(
    model,
    word: str,
    num_batches: int,
    tokenizer,
    loader,
    device: str = "cpu",
    save_to: Optional[str] = None,
) -> torch.Tensor:
    """
    Collects hidden states for each occurrence of `word`.

    Returns:
      all_hiddens: Tensor of shape [num_layers, total_occurrences, hidden_dim]
    """
    token_id: int = tokenizer(word, add_special_tokens=False)["input_ids"][0]
    all_hiddens: List[torch.Tensor] = []
    counter = 0

    for batch in tqdm(loader):
        if counter >= num_batches:
            break

        ids = batch["input_ids"].to(device)
        batch_idx, pos_idx = torch.where(ids == token_id)
        if batch_idx.numel() == 0:
            continue

        counter += 1
        with torch.no_grad():
            out = model(ids, output_hidden_states=True)
        hidden_states = out.hidden_states
        layer_states = hidden_states[1:]

        per_layer_vecs = []
        for layer in layer_states:
            vecs = layer[batch_idx, pos_idx]
            per_layer_vecs.append(vecs.cpu())

        batch_hiddens = torch.stack(per_layer_vecs, dim=0)
        all_hiddens.append(batch_hiddens)

    all_hiddens_tensor = torch.cat(all_hiddens, dim=1)
    if save_to:
        torch.save(all_hiddens_tensor, save_to)

    return all_hiddens_tensor


def gather_same_values(
    model,
    word: str,
    num_batches: int,
    tokenizer,
    loader,
    device: str = "cpu",
    save_to: Optional[str] = None,
) -> torch.Tensor:
    """
    Collects attention value states for each occurrence of `word`.

    Returns:
      all_values: Tensor [num_layers, total_occurrences, hidden_dim]
    """
    token_id = tokenizer(word, add_special_tokens=False)["input_ids"][0]
    num_layers = model.config.num_hidden_layers
    all_values = []
    counter = 0

    for batch in tqdm(loader):
        if counter >= num_batches:
            break

        ids = batch["input_ids"].to(device)
        batch_idx, pos_idx = torch.where(ids == token_id)
        if batch_idx.numel() == 0:
            continue

        counter += 1
        with torch.no_grad():
            out = model(ids)

        per_layer_vals: List[torch.Tensor] = []
        for layer_idx in range(num_layers):
            vals = model.model.layers[layer_idx].self_attn.value_states
            vals = rearrange(vals, "b h t d -> b t (h d)")
            vecs = vals[batch_idx, pos_idx].detach().cpu()
            per_layer_vals.append(vecs)

        batch_values = torch.stack(per_layer_vals, dim=0)
        all_values.append(batch_values)

    all_values_tensor = torch.cat(all_values, dim=1)
    if save_to:
        torch.save(all_values_tensor, save_to)

    return all_values_tensor


def hiddens_tsne(
    all_hiddens: Dict[str, Dict[str, torch.Tensor]],
    count: Optional[int] = None,
) -> Dict[str, np.ndarray]:
    """
    Runs t-SNE on each (model, word, layer) slice.
    """
    tsne_results: Dict[str, any] = {}
    for model_name, word_map in all_hiddens.items():
        num_layers = next(iter(word_map.values())).shape[0]
        for layer_idx in range(num_layers):
            # Gather for all words
            slices = []
            for word, tensor in word_map.items():
                arr = tensor[layer_idx]
                if count is not None:
                    arr = arr[:count]
                slices.append(arr)
            combined = torch.cat(slices, dim=0).numpy()
            normalized = normalize(combined)
            emb2d = TSNE(
                n_components=2, random_state=42, max_iter=1000, n_jobs=-1
            ).fit_transform(normalized)

            # split back per word
            start = 0
            for word, arr in zip(word_map.keys(), slices):
                length = arr.shape[0]
                key = f"{model_name}_{word.strip()}_layer{layer_idx}"
                tsne_results[key] = emb2d[start : start + length]
                start += length
    return tsne_results
