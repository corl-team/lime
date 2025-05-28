import gc
from collections import defaultdict

import numpy as np
import torch
from tqdm.autonotebook import tqdm


@torch.no_grad()
def get_dataset_statistics(hiddens):
    return {
        "mean": hiddens.mean(dim=(0, 1), keepdim=True),
        "std": hiddens.std(dim=(0, 1), keepdim=True),
        "norm": torch.norm(hiddens, p=2, dim=-1, keepdim=True),
    }


@torch.no_grad()
def shannon_matrix_entropy(hidden, alpha=0.99, eps=1e-9):
    k = hidden @ hidden.T
    assert torch.all(torch.eq(k, k.T)), k
    trace = torch.trace(k)
    eig = torch.linalg.eigvalsh(k)
    eig.clamp_(min=0)
    assert torch.all(eig >= 0), (eig < 0).sum()
    assert torch.float64 == eig.dtype, eig.dtype
    z = (eig / trace) ** alpha
    z = z.sum()
    ent = torch.log(z) / (1 - alpha)
    return ent


@torch.no_grad()
def get_hiddens(model, loader, num_samples, device="cuda"):
    torch.cuda.empty_cache()
    gc.collect()
    hiddens = defaultdict(list)

    count = 0
    for batch in loader:
        if count >= num_samples:
            break
        batch = {k: v.to(device) for k, v in batch.items()}
        count += batch["input_ids"].shape[0]

        hidden_states = model(
            batch["input_ids"], output_hidden_states=True
        ).hidden_states
        hidden_states = (
            hidden_states[:-1]
            + (model.model.norm.hidden_before_norm,)
            + (hidden_states[-1],)
        )
        hidden_states = (hs.cpu() for hs in hidden_states)

        for i, hs in enumerate(hidden_states):
            hiddens[i] += [hs]

    hiddens = {k: torch.vstack(v) for k, v in hiddens.items()}
    return hiddens


@torch.no_grad()
def get_values(model, loader, num_samples, device="cuda"):
    torch.cuda.empty_cache()
    gc.collect()
    hiddens = defaultdict(list)

    count = 0
    for batch in loader:
        if count >= num_samples:
            break
        batch = {k: v.to(device) for k, v in batch.items()}
        count += batch["input_ids"].shape[0]

        outs = model(batch["input_ids"], output_hidden_states=False)
        cur_values = []
        for layer_idx in range(model.model.config.num_hidden_layers):
            cur_values += [model.model.layers[layer_idx].self_attn.value_states.cpu()]

        for i, hs in enumerate(cur_values):
            hiddens[i] += [hs]

    hiddens = {k: torch.vstack(v) for k, v in hiddens.items()}
    return hiddens


@torch.no_grad()
def calculate_entropy(
    hs, num_samples, b, alpha=0.5, eps=1e-5, values=False, device="cuda"
):
    idx = int(not values)

    entropies = []
    stds = []
    for i in tqdm(range(idx, 16 + idx)):
        hs[i] = hs[i].to(torch.float64)
        dataset_statistics = get_dataset_statistics(hs[i])
        normalized_hs = (hs[i] - dataset_statistics["mean"]) / (
            dataset_statistics["std"] + eps
        )
        entropies_layer = []
        for sample_idx in range(num_samples):
            hidden = normalized_hs[sample_idx]
            entropies_layer += [
                shannon_matrix_entropy(hidden.to(device), alpha=alpha, eps=eps).item()
            ]

        entropies += [np.mean(entropies_layer)]
        stds += [np.std(entropies_layer)]

    return np.array(entropies), np.array(stds)
