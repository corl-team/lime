import torch
from numpy import save
from safetensors.torch import load_file, load_model
from transformers import LlamaConfig, LlamaForCausalLM

from config import HCConfig, LIMeConfig, ModelConfig
from datasets import load_from_disk

from src.lm.hc import LlamaHCForCausalLM
from src.lm.lime import LIMeForCausalLM
from src.analysis.values_gathering_wrapper import value_wrapper


class HookNorm(torch.nn.Module):
    def __init__(self, norm_class):
        super().__init__()
        self.norm_class = norm_class

    def forward(self, x):
        self.hidden_before_norm = x
        return self.norm_class(x)


def get_dataloader(num_samples, batch_size, dataset_path):
    dataset = load_from_disk(dataset_path).select(range(num_samples))
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False)
    return loader


def get_model(setup, save_values=False, top_p=None, save_unnormed_hs=True, path=None):
    """
    Create configs and load model weights.
    If `save_values` is True, each `self_attn` module will have attribute `value_states`.
    If `save_unnormed_hs` is True, last hidden state will be added to `hidden_states` before last layer norm.
    """
    print(f"setup: {setup}")
    if setup == "llama":
        configs = create_configs(model_type="llama")
        model = load_pretrained_model(
            run_name="llama",
            model_type="llama",
            model_config=configs["model_config"],
            lime_config=None,
            hc_config=None,
            save_values=save_values,
            path=path,
        )
    elif setup == "lime_static":
        configs = create_configs(model_type="lime_static", top_p=top_p)
        model = load_pretrained_model(
            run_name="lime_static",
            model_type="lime",
            model_config=configs["model_config"],
            lime_config=configs["lime_config"],
            hc_config=None,
            save_values=save_values,
            path=path,
        )
    elif setup == "lime_dynamic":
        configs = create_configs(model_type="lime_dynamic")
        model = load_pretrained_model(
            run_name="lime_dynamic",
            model_type="lime",
            model_config=configs["model_config"],
            lime_config=configs["lime_config"],
            hc_config=None,
            save_values=save_values,
            path=path,
        )
    elif setup == "hc":
        hc_config = HCConfig()
        configs = create_configs(model_type="hc")
        model = load_pretrained_model(
            run_name="hc",
            model_type="hc",
            model_config=configs["model_config"],
            hc_config=hc_config,
            save_values=save_values,
            path=path,
        )
    if save_unnormed_hs:
        model.model.norm = HookNorm(model.model.norm)

    return model


def add_missing_params(state_dict, missing_keys, llama_config):
    """
    Add missing parameters to the state_dict.
    """
    for key in missing_keys:
        if "top_p_weights" in key:
            idx = int(key.split(".")[2]) + 1
            tensor = torch.zeros(
                (llama_config.num_attention_heads, idx), requires_grad=False
            )
            state_dict[key] = tensor.detach()
    state_dict["lm_head.weight"] = state_dict["model.embed_tokens.weight"]
    return state_dict


def create_configs(model_type: str, top_p: float = None):
    configs = {}
    model_config = ModelConfig()
    model_config = LlamaConfig(
        vocab_size=model_config.vocab_size,
        hidden_size=2048,
        intermediate_size=2048 * 4,
        max_position_embeddings=2048,
        num_key_value_heads=32,
        num_attention_heads=32,
        num_hidden_layers=16,
        tie_word_embeddings=True,
        use_cache=False,
    )
    configs["model_config"] = model_config
    if model_type == "llama":
        return configs

    elif model_type == "lime_static":
        lime_config = LIMeConfig(dynamic=False, top_p=top_p, descending=False)
        configs["lime_config"] = lime_config
        return configs

    elif model_type == "lime_dynamic":
        lime_config = LIMeConfig(dynamic=True)
        configs["lime_config"] = lime_config
        return configs

    elif model_type == "hc":
        configs["hc_config"] = HCConfig()
        return configs
    else:
        raise NotImplementedError(f"model type {model_type} is not supported")


def load_pretrained_model(
    run_name,
    model_type="llama",
    model_config=None,
    lime_config=None,
    hc_config=None,
    save_values=False,
    path=None,
):
    if model_type.startswith("lime"):
        model = LIMeForCausalLM(model_config, lime_config)
    elif model_type == "hc":
        model = LlamaHCForCausalLM(model_config, hc_config)
    else:
        model = LlamaForCausalLM(model_config)

    if save_values:
        model = value_wrapper(model, model_type)
    art_path = "../artifacts/"
    full_path = path or art_path + run_name

    if model_type.startswith("lime") and lime_config.top_p is not None:
        state_dict = load_file(f"{full_path}/model.safetensors")
        buffers = ["top_p_weights"]
        missing_keys = []
        for layer_idx in range(1, model_config.num_hidden_layers):
            for buffer in buffers:
                missing_keys += [f"model.layers.{layer_idx}.attention_router.{buffer}"]
        state_dict = add_missing_params(state_dict, missing_keys, model_config)
        model.load_state_dict(state_dict)
    else:
        missing, unexpected = load_model(
            model, f"{full_path}/model.safetensors", strict=True
        )
        print("Missing keys:", missing)
        print("Unexpected keys:", unexpected)
        assert len(unexpected) == 0
        assert len(missing) == 0

    return model


def calculate_head_norms(attention_layer):
    """
    Extracts the weights of W_o and calculates the norm per head.
    """
    W_o = attention_layer.o_proj.weight.mT

    hidden_dim = W_o.shape[0]
    num_heads = attention_layer.num_heads
    head_dim = hidden_dim // num_heads
    W_o_per_head = W_o.view(num_heads, head_dim, hidden_dim)

    head_norms = torch.norm(W_o_per_head, dim=(1, 2))

    return head_norms
