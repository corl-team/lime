import os.path
import time
from dataclasses import asdict
from datetime import datetime, timedelta
from functools import partial
from random import seed as py_seed
from typing import Any, Optional, Union

import accelerate
import pyrallis
import torch
import torch.distributed
import wandb
from accelerate import (
    AutocastKwargs,
    DataLoaderConfiguration,
    DistributedDataParallelKwargs,
    InitProcessGroupKwargs,
)
from einops import repeat
from lion_pytorch import Lion
from numpy.random import seed as np_seed
from torch import FloatTensor, Tensor
from torch.nn import functional as F
from torch.optim import lr_scheduler
from transformers import LlamaConfig, LlamaForCausalLM
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.utils import is_flash_attn_2_available

from config import TrainConfig
from datasets import load_from_disk
from meters import ProgressMeter
from src.lm.hc import LlamaHCForCausalLM
from src.lm.lime import LIMeForCausalLM
from src.utils.flops_counter import get_model_flops
from train_utils import (
    cast_embedding_output_to_autocast_dtype,
    configure_optimizers,
    count_parameters,
    fix_llama_rmsnorm_cast,
    get_cosine_schedule_with_warmup,
)


def train_step(
    batch,
    model,
    optimizer,
    scheduler,
    accelerator: accelerate.Accelerator,
    config: TrainConfig,
    step=1,
) -> tuple[dict[str, Union[Optional[FloatTensor], Any]], Optional[Tensor]]:
    input_ids, labels = batch['input_ids'], batch['labels']
    
    if config.debug:
        accelerator.print("Forward pass...")
    # print("start forward")
    outputs = model(
        input_ids=input_ids, output_hidden_states=False
    )

    logits = outputs.logits.float()
    loss_fct = torch.nn.CrossEntropyLoss()
    logits = logits.view(-1, config.model_config.vocab_size)
    labels = labels.view(-1)
    labels = labels.to(logits.device)
    loss = loss_fct(logits, labels)

    if config.debug:
        accelerator.print("Backward pass...")
    with torch.no_grad():
        preds = outputs.logits.argmax(-1).view(-1)
        labels_acc = labels.view(-1)
        accuracy = (preds == labels_acc).sum() / labels_acc.size(0)
    
    accelerator.backward(loss)
    grad_norm = None
    if accelerator.sync_gradients:
        grad_norm = accelerator.clip_grad_norm_(
            model.parameters(), config.max_grad_norm
        )
    optimizer.step()
    scheduler.step()
    optimizer.zero_grad()

    metrics = {
        "loss": loss.detach(),
        "accuracy": accuracy,
    }


    if config.debug:
        accelerator.print(metrics)
    return metrics, grad_norm


def train(
    dataloader,
    model,
    optimizer,
    scheduler,
    accelerator: accelerate.Accelerator,
    config: TrainConfig,
):
    if accelerator.is_main_process:
        meters_names = ["loss", "data_time", "model_time", "accuracy", "lr"]
        progress = ProgressMeter(config.num_steps, meter_names=meters_names)

    accumulated_metrics = None
    model.train()

    min_loss = float('inf')
    max_accuracy = -float('inf')

    end = time.time()
    for idx, batch in enumerate(dataloader):
        start = time.time()
        step = idx // accelerator.gradient_accumulation_steps + 1
        if accelerator.is_main_process:
            batch_size = batch["input_ids"].shape[0]
            data_time = time.time() - end

        with accelerator.accumulate(model):
            output_dict, grad_norm = train_step(
                batch, model, optimizer, scheduler, accelerator, config, step
            )

        # gathering metrics among all grad accum steps
        if accumulated_metrics is None:
            accumulated_metrics = output_dict
        else:
            for metric, val in output_dict.items():
                accumulated_metrics[metric] += val
        
        model_time = time.time() - start

        if accelerator.sync_gradients:
            accumulated_metrics = accelerator.reduce(accumulated_metrics, reduction="mean")

            if accelerator.is_main_process:
                accumulated_metrics = {
                    k: v.detach().item() / config.gradient_accumulation_steps if isinstance(v, torch.Tensor) else v / config.gradient_accumulation_steps
                    for k, v in accumulated_metrics.items()
                }
                
                accumulated_metrics["lr"] = scheduler.get_last_lr()[0]
                accumulated_metrics["data_time"] = data_time
                accumulated_metrics["model_time"] = model_time
                num_samples = batch_size * config.gradient_accumulation_steps * accelerator.num_processes
                progress.update(n=num_samples, **{k: v for k, v in accumulated_metrics.items()})

                # log summary
                if accumulated_metrics['loss'] < min_loss:
                    accelerator.trackers[0].tracker.summary['min_loss'] = accumulated_metrics['loss']
                if accumulated_metrics['accuracy'] > max_accuracy:
                    accelerator.trackers[0].tracker.summary['max_acc'] = accumulated_metrics['accuracy']                


                accumulated_metrics["grad_norm"] = grad_norm
                accelerator.log(accumulated_metrics, step=step)
                if accelerator.is_main_process:
                    progress.display(step)
            
            accumulated_metrics = None

            if step == config.num_steps:
                break
            end = time.time()
    accelerator.save_state(config.save_path + "/final")
    if accelerator.is_main_process:
        artifact = wandb.Artifact(name="training_state_final", type="training_state")
        artifact.add_dir(config.save_path + f"/final")
        wandb.log_artifact(artifact)
    accelerator.wait_for_everyone()


@pyrallis.wrap()
def main_train(config: TrainConfig):
    torch.manual_seed(config.seed)
    torch.cuda.manual_seed(config.seed)
    py_seed(config.seed)
    np_seed(config.seed)

    autocast_kwargs = AutocastKwargs(enabled=True, cache_enabled=True)
    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=False)
    init_process_kwargs = InitProcessGroupKwargs(timeout=timedelta(seconds=3600))
    dataloader_config = DataLoaderConfiguration(split_batches=True)

    run_name = config.wandb_config.run_name
    config.save_path += f'{run_name}_{datetime.now().isoformat(sep="_", timespec="seconds")}'
    print(config.save_path)

    if config.enable_fsdp:
        from accelerate import FullyShardedDataParallelPlugin
        from torch.distributed.fsdp.fully_sharded_data_parallel import (
            FullOptimStateDictConfig,
            FullStateDictConfig,
        )

        fsdp_plugin = FullyShardedDataParallelPlugin(
            auto_wrap_policy="TRANSFORMER_BASED_WRAP",
            state_dict_config=FullStateDictConfig(
                offload_to_cpu=False, rank0_only=False
            ),
            optim_state_dict_config=FullOptimStateDictConfig(
                offload_to_cpu=False, rank0_only=False
            ),
        )
    else:
        fsdp_plugin = None

    accelerator = accelerate.Accelerator(
        mixed_precision=config.mixed_precision,
        log_with="wandb",
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        kwargs_handlers=[autocast_kwargs, ddp_kwargs, init_process_kwargs],
        dynamo_backend=config.dynamo_backend,
        step_scheduler_with_optimizer=True,
        dataloader_config=dataloader_config,
        fsdp_plugin=fsdp_plugin,
    )

    accelerator.init_trackers(
        project_name=(
            config.wandb_config.project
            if not config.debug
            else "debug_" + config.wandb_config.project
        ),
        config=vars(config),
        init_kwargs={"wandb": {"entity": config.wandb_config.entity, "name": run_name, "allow_val_change": True}},
    )

    accelerator.print("[*] Using automatic precision for matmul kernels 💡 [*]")

    torch.backends.cuda.matmul.allow_tf32 = True # allow tf32 on matmul
    torch.backends.cudnn.allow_tf32 = True # allow tf32 on cudnn
    torch.set_float32_matmul_precision("high")

    accelerator.print("[*] Loading data 📚... [*]")
    if config.debug:
        with accelerator.main_process_first():
            dataset = load_from_disk(config.data_config.data_path).select(range(10240))
    else:
        with accelerator.main_process_first():
            num_samples = config.num_steps * config.effective_batch_size
            dataset = load_from_disk(config.data_config.data_path).select(range(num_samples))
    
    if accelerator.is_main_process:
        print("Dataset size:", len(dataset))

    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=config.effective_batch_size // config.gradient_accumulation_steps,
        shuffle=False,
        num_workers=config.data_config.num_workers,
    )
    dataloader = accelerator.prepare_data_loader(dataloader)
    accelerator.print("[*] Success! Data is loaded 📖 [*] ")
    accelerator.print("[*] Initializing model 🤖 [*] ")
    if is_flash_attn_2_available():
        accelerator.print("⚡ Using Flash Attention for training ⚡")
    else:
        accelerator.print("Flash attention is not available 😠")
    
    with accelerator.main_process_first():
        model_config = LlamaConfig(
            vocab_size=config.model_config.vocab_size,
            hidden_size=config.model_config.hidden_size,
            intermediate_size=config.model_config.intermediate_size,
            max_position_embeddings=config.data_config.seq_length,
            num_key_value_heads=config.model_config.num_attention_heads, # set automatically to num_attention_heads
            num_attention_heads=config.model_config.num_attention_heads,
            num_hidden_layers=config.model_config.num_hidden_layers,
            tie_word_embeddings=config.model_config.tie_word_embeddings,
            use_cache=config.model_config.use_cache,
        )
        if accelerator.is_main_process:
            model_config.save_pretrained(config.save_path + "/config")
            artifact = wandb.Artifact("model_config", "config")
            artifact.add_dir(config.save_path + "/config")
            wandb.log_artifact(artifact)

            
        if config.model_type == 'llama':
            model_cls = LlamaForCausalLM
            model = model_cls(model_config)

        elif config.model_type == 'lime':
            model_cls = LIMeForCausalLM
            model = model_cls(model_config, config.lime_config)

        elif config.model_type == 'hc':
            model_cls = LlamaHCForCausalLM
            model = model_cls(model_config, config.hc_config)
                

        if config.model_config.tie_word_embeddings:
            model.tie_weights()

        # autocast does not cast embeddings to mp automatically
        model.model.embed_tokens.register_forward_hook(cast_embedding_output_to_autocast_dtype)
        # original llama code has wrong casting in rmsnorm
        fix_llama_rmsnorm_cast(model)

    
    if accelerator.is_main_process:
        with torch.no_grad():
            forward_flops = get_model_flops(model)
        accelerator.log({"forward_flops": forward_flops}, step=1)
    accelerator.wait_for_everyone()

    if accelerator.is_main_process:
        count_parameters(model)
        total_params = sum(p.numel() for p in model.parameters())
        wandb.config.update({"parameters_count": total_params})
        wandb.config.update(asdict(config))
        print(model)

    model = accelerator.prepare_model(model)

    accelerator.print("[*] Model successfully initialized 🦾 [*] ")
    accelerator.print("[*] Initializing optimizer 🏂🏻 and scheduler [*]")
    optim_groups = configure_optimizers(accelerator.unwrap_model(model), config, accelerator.is_main_process)

    if config.optimizer == "AdamW":
        optimizer = torch.optim.AdamW(
            params=optim_groups,
            lr=config.lr,
            fused=True if torch.cuda.is_available() else None,
            betas=(config.beta1, config.beta2),
            eps=config.eps
        )
    elif config.optimizer == "Lion":
        optimizer = Lion(params=optim_groups, lr=config.lr, use_triton=True)
    elif config.optimizer == "SGD":
        optimizer = torch.optim.SGD(
            params=optim_groups,
            lr=config.lr,
            momentum=0.9
        )
    else:
        raise ValueError(f"Optimizer {config.optimizer} is not understood")
    
    scheduler = get_cosine_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=config.num_warmup_steps,
        num_training_steps=config.num_steps,
        min_lr=config.min_lr,
    )

    optimizer, scheduler = accelerator.prepare(optimizer, scheduler)
    accelerator.print("[*] Done ⛰️ , ready for the descent![*]")
    if os.path.exists(config.save_path):
        pass
    accelerator.print("[*] Starting training 👨‍🏫 [*]")
    train(dataloader, model, optimizer, scheduler, accelerator, config)
    accelerator.print("[*]Done! 👩‍🎓🎓🎓🎓🎓[*]")
    accelerator.end_training()
    accelerator.print(f"Model saved to {config.save_path}")


if __name__ == "__main__":
    main_train()
