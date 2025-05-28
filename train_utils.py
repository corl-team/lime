import math
from functools import partial

import torch
from prettytable import PrettyTable
from torch.optim.lr_scheduler import LambdaLR
from transformers.models.llama.modeling_llama import LlamaForCausalLM, LlamaRMSNorm

from config import TrainConfig


def _rmsnorm_custom_forward(self, hidden_states):
    input_dtype = hidden_states.dtype
    hidden_states = hidden_states.to(torch.float32)
    variance = hidden_states.pow(2).mean(-1, keepdim=True)
    hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
    return (self.weight.to(torch.float32) * hidden_states).to(input_dtype)

def _recursive_rmsnorm_search(module):
    if isinstance(module, LlamaRMSNorm):
        module.forward = _rmsnorm_custom_forward.__get__(module, LlamaRMSNorm)

    for child in module.children():
        _recursive_rmsnorm_search(child)

def fix_llama_rmsnorm_cast(model: LlamaForCausalLM):
    _recursive_rmsnorm_search(model)


def _cast_if_autocast_enabled(tensor):
    if torch.is_autocast_enabled():
        if tensor.device.type == 'cuda':
            dtype = torch.get_autocast_gpu_dtype()
        elif tensor.device.type == 'cpu':
            dtype = torch.get_autocast_cpu_dtype()
        else:
            raise NotImplementedError()
        return tensor.to(dtype=dtype)
    return tensor

def cast_embedding_output_to_autocast_dtype(module, input, output):
    return _cast_if_autocast_enabled(output)


def count_parameters(model):
    table = PrettyTable(["Modules", "Parameters"])
    total_params = 0
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        params = parameter.numel()
        table.add_row([name, params])
        total_params += params
    print(table)
    print(f"Total Trainable Params: {total_params}")
    return total_params

def configure_optimizers(model, train_config: TrainConfig, is_main: bool):
    # from https://github.com/karpathy/minGPT/blob/37baab71b9abea1b76ab957409a1cc2fbfba8a26/mingpt/model.py#L254
    """
    This long function is unfortunately doing something very simple and is being very defensive:
    We are separating out all parameters of the model into two buckets: those that will experience
    weight decay for regularization and those that won't (biases, and layernorm/embedding weights).
    We are then returning the PyTorch optimizer object.
    """

    # separate out all parameters to those that will and won't experience regularizing weight decay
    decay = set()
    no_decay = set()

    large_lr_router_params = set()
    for mn, m in model.named_modules():
        for pn, p in m.named_parameters():
            fpn = "%s.%s" % (mn, pn) if mn else pn
            if pn.endswith('static_weights') or 'dynamic_proj' in pn:
                large_lr_router_params.add(fpn)
    print('large_lr_router_params', large_lr_router_params)

    whitelist_weight_modules = (torch.nn.Linear)
    blacklist_weight_modules = (torch.nn.LayerNorm, torch.nn.Embedding, LlamaRMSNorm)

    hc_params_static = ['static_beta',
                        'static_alpha']
    hc_params_dynamic = ['dynamic_alpha_fn',
                         'dynamic_alpha_scale',
                         'dynamic_beta_fn',
                         'dynamic_beta_scale']
    
    lime_params_no_decay = [] if train_config.lime_config.wd_on_static_weights else ['static_weights']
    lime_params_decay = ['static_weights', 'dynamic_proj'] if train_config.lime_config.wd_on_static_weights else ['dynamic_proj']

    for mn, m in model.named_modules():
        for pn, p in m.named_parameters():
            fpn = "%s.%s" % (mn, pn) if mn else pn  # full param name
            # random note: because named_modules and named_parameters are recursive
            # we will see the same tensors p many many times. but doing it this way
            # allows us to know which parent module any tensor p belongs to...
            if pn.endswith("bias") or pn in hc_params_static or pn in lime_params_no_decay:
                no_decay.add(fpn)
            elif pn.endswith("weight") and isinstance(m, whitelist_weight_modules) or pn in lime_params_decay:
                # weights of whitelist modules will be weight decayed
                decay.add(fpn)
            elif pn.endswith('lime_kv_proj'):
                decay.add(fpn)
            elif pn.endswith("weight") and isinstance(m, blacklist_weight_modules):
                # weights of blacklist modules will NOT be weight decayed
                no_decay.add(fpn)
            elif pn in hc_params_dynamic:
                decay.add(fpn)
    
    if train_config.model_config.tie_word_embeddings:
        if '_orig_mod.lm_head.weight' not in decay:
            decay.remove('lm_head.weight')
        else:
            decay.remove('_orig_mod.lm_head.weight')

    # validate that we considered every parameter
    param_dict = {pn: p for pn, p in model.named_parameters()}
    inter_params = decay & no_decay
    union_params = decay | no_decay
    assert (
        len(inter_params) == 0
    ), "parameters %s made it into both decay/no_decay sets!" % (str(inter_params),)
    assert (
        len(param_dict.keys() - union_params) == 0
    ), "parameters %s were not separated into either decay/no_decay/repeat_classifier set!" % (
        str(param_dict.keys() - union_params),
    )

    if is_main:
        print('=' * 50)
        print("DECAY")
        print(sorted(list(decay)))
        print('=' * 50)
        print("NO DECAY")
        print(sorted(list(no_decay)))
        print("LARGE LR DECAY")
        print(sorted(list(large_lr_router_params & decay)))
        print('=' * 50)
        print("LARGE LR NO DECAY")
        print(sorted(list(large_lr_router_params & no_decay)))
    
    # create the pytorch optimizer object

    optim_groups = [
        {
            "params": [param_dict[pn] for pn in sorted(list(decay)) if pn in param_dict and pn not in large_lr_router_params],
            "weight_decay": train_config.weight_decay,
        },
        {
            "params": [param_dict[pn] for pn in sorted(list(no_decay)) if pn in param_dict and pn not in large_lr_router_params],
            "weight_decay": 0.0,
        },

        {
            "params": [param_dict[pn] for pn in sorted(list(no_decay)) if pn in param_dict and pn in large_lr_router_params],
            "weight_decay": 0.0,
            "lr": train_config.lime_config.router_lr,
        },

        {
            "params": [param_dict[pn] for pn in sorted(list(decay)) if pn in param_dict and pn in large_lr_router_params],
            "weight_decay": train_config.weight_decay,
            "lr": train_config.lime_config.router_lr,
        },
    ]
    return optim_groups

def _get_cosine_schedule_with_warmup_lr_lambda(
    current_step: int,
    *,
    num_warmup_steps: int,
    num_training_steps: int,
    num_cycles: float,
    min_lr: float = 0.0
):
    if current_step < num_warmup_steps:
        return float(current_step) / float(max(1, num_warmup_steps))
    progress = float(current_step - num_warmup_steps) / float(
        max(1, num_training_steps - num_warmup_steps)
    )
    return max(
        min_lr, 0.5 * (1.0 + math.cos(math.pi * float(num_cycles) * 2.0 * progress))
    )


def get_cosine_schedule_with_warmup(
    optimizer,
    num_warmup_steps: int,
    num_training_steps: int,
    num_cycles: float = 0.5,
    min_lr: float = 0.0,
    last_epoch: int = -1,
):
    """
    Create a schedule with a learning rate that decreases following the values of the cosine function between the
    initial lr set in the optimizer to 0, after a warmup period during which it increases linearly between 0 and the
    initial lr set in the optimizer.

    Args:
        optimizer ([`~torch.optim.Optimizer`]):
            The optimizer for which to schedule the learning rate.
        num_warmup_steps (`int`):
            The number of steps for the warmup phase.
        num_training_steps (`int`):
            The total number of training steps.
        num_cycles (`float`, *optional*, defaults to 0.5):
            The number of waves in the cosine schedule (the defaults is to just decrease from the max value to 0
            following a half-cosine).
        last_epoch (`int`, *optional*, defaults to -1):
            The index of the last epoch when resuming training.

    Return:
        `torch.optim.lr_scheduler.LambdaLR` with the appropriate schedule.
    """

    lr_lambda = partial(
        _get_cosine_schedule_with_warmup_lr_lambda,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps,
        num_cycles=num_cycles,
        min_lr=min_lr,
    )
    return LambdaLR(optimizer, lr_lambda, last_epoch)
