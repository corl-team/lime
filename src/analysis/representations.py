from json import load
from typing import Dict, Optional

import torch
from sklearn.manifold import TSNE
from torch.nn import functional as F
from torch.nn.functional import normalize as norm
from tqdm.autonotebook import tqdm


def gather_same_hiddens(
    model, word, num_batches, tokenizer, loader, device="cpu", save_to=None
):
    """
    Gather hidden states from a model for a specific word across multiple batches.
    """
    token_id = tokenizer(word)["input_ids"][0]

    all_word_hiddens = []
    counter = 0
    for batch in tqdm(loader):
        if counter >= num_batches:
            break
        tok_batch = tokenizer(
            batch["text"], return_tensors="pt", truncation=True, padding=True
        )
        word_indices = torch.where(tok_batch["input_ids"] == token_id)
        if word_indices[0].shape[0] == 0:
            continue

        counter += 1
        tok_batch = {k: v.to(device) for k, v in tok_batch.items()}
        with torch.no_grad():
            hs = model(tok_batch["input_ids"], output_hidden_states=True).hidden_states
            hs = hs[:-1] + (model.model.norm.hidden_before_norm,) + (hs[-1],)

        all_word_hiddens += [
            torch.stack([hs[i][word_indices] for i in range(len(hs))]).cpu()
        ]

    all_word_hiddens = torch.cat(all_word_hiddens, dim=1)
    if save_to is not None:
        torch.save(all_word_hiddens, save_to)

    return all_word_hiddens


def gather_same_values(
    model, word, num_batches, tokenizer, loader, device="cpu", save_to=None
):
    """
    Gather value states from a model for a specific word across multiple batches.
    """
    token_id = tokenizer(word)["input_ids"][0]

    all_values = []
    counter = 0
    for batch in tqdm(loader):
        if counter >= num_batches:
            break
        tok_batch = tokenizer(
            batch["text"], return_tensors="pt", truncation=True, padding=True
        )
        word_indices = torch.where(tok_batch["input_ids"] == token_id)
        if word_indices[0].shape[0] == 0:
            continue

        counter += 1
        tok_batch = {k: v.to(device) for k, v in tok_batch.items()}
        with torch.no_grad():
            outs = model(tok_batch["input_ids"])
        hs = []
        for layer_idx in range(model.model.config.num_hidden_layers):
            hs += [model.model.layers[layer_idx].self_attn.value_states.detach()]

        all_values += [torch.stack([hs[i][word_indices] for i in range(len(hs))]).cpu()]

    all_values = torch.cat(all_values, dim=1)
    if save_to is not None:
        torch.save(all_values, save_to)

    return all_values


def gather_all_representations(
    model,
    words,
    num_batches,
    tokenizer,
    loader,
    device="cpu",
    values=False,
    save_to=None,
):
    """
    Gather all representations (either hidden states or value states) for a list of words.
    """
    all_representations = {}
    min_amount = float("inf")
    gather_func = gather_same_values if values else gather_same_hiddens
    for word in words:
        all_representations[word] = gather_func(
            model, word, num_batches, tokenizer, loader, device, save_to=None
        )
        min_amount = min(min_amount, all_representations[word].shape[1])

    # Cut off to have the same amount of representations for each word
    for word in words:
        all_representations[word] = all_representations[word][:, :min_amount, :]

    if save_to is not None:
        torch.save(all_representations, save_to)

    return all_representations


def hiddens_tsne(
    all_hiddens: Dict[str, Dict[str, torch.Tensor]],
    num_layers: int = 16,
    count: Optional[int] = None,
):
    """
    all_hiddens are in format  {'model_name': {'word': representations, ...}, ...},
    where representations is a torch tensor of shape [num_layers, count]
    """
    all_hiddens_tsne = dict()
    for layer_idx in tqdm(range(num_layers)):
        for model_name, model_hiddens in all_hiddens.items():
            tsne = TSNE(n_components=2, random_state=42, max_iter=1000, n_jobs=-1)
            hsw = []
            for word, hs in model_hiddens.items():
                hsw += [all_hiddens[model_name][word][layer_idx][:count]]
            cur_h = torch.cat(hsw, dim=0)

            for i, word in enumerate(list(list(all_hiddens.values())[0].keys())):
                all_hiddens_tsne[f"{word.strip()}_{model_name}_{layer_idx}"] = (
                    tsne.fit_transform(
                        F.layer_norm(norm(cur_h, dim=-1), normalized_shape=cur_h.shape)
                    )[count * i : count * (i + 1)]
                )

    return all_hiddens_tsne
