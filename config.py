import os
from dataclasses import dataclass, field


@dataclass
class DataConfig:
    data_path: str = None
    start_idx: int = 0
    seq_length: int = 2048
    num_workers: int = 1
    warmup_dataset: bool = False


@dataclass
class WandbConfig:
    project: str
    entity: str = None
    run_name: str = None
    save_checkpoint_to_wandb: bool = False


@dataclass
class ModelConfig:
    vocab_size: int = 50257
    hidden_size: int = 2048
    intermediate_size: int = 8192
    tie_word_embeddings: bool = True
    num_hidden_layers: int = 16
    num_attention_heads: int = 32
    use_flash_attention: bool = False
    use_cache: bool = False
    num_key_value_heads: int = None


@dataclass
class LIMeConfig:
    router_lr: float = 1e-2


@dataclass
class HCConfig:
    dynamic: bool = True
    rate: int = 4
    norm: str = "none"


@dataclass
class TrainConfig:
    save_path: str
    num_steps: int
    data_config: DataConfig
    wandb_config: WandbConfig
    dynamo_backend: str = "inductor"
    model_config: ModelConfig = field(default_factory=ModelConfig)
    lime_config: LIMeConfig = field(default_factory=LIMeConfig)
    hc_config: HCConfig = field(default_factory=HCConfig)
    debug: bool = False
    resume_training: bool = False
    lr: float = 0.001
    weight_decay: float = 0.1
    beta1: float = 0.9
    beta2: float = 0.95
    eps: float = 1e-08
    num_warmup_steps: int = 200
    optimizer: str = "AdamW"
    mixed_precision: str = "bf16"
    scheduler: str = "cosine"
    save_model_interval: int = 2500
    log_interval: int = 10
    eval_interval: int = None
    gradient_accumulation_steps: int = 1
    effective_batch_size: int = 1024
    batch_size_per_device: int = None
    eval_batch_size: int = None
    max_grad_norm: float = 1.0
    min_lr: float = 1e-6
    enable_fsdp: bool = False
    model_name: str = None
    pretrained: str = None
    seed: int = 24
    wrapper_patch: bool = False
    model_type: str = "llama"
    pretrained_path: str = None
    num_eval_steps: int = 4

    def __post_init__(self):

        assert self.batch_size_per_device is not None
        if self.eval_interval is not None:
            assert self.eval_interval % self.log_interval == 0

        assert self.dynamo_backend in ("no", "inductor")

        num_processes = int(os.getenv("WORLD_SIZE", 1))
        if (
            self.effective_batch_size == self.batch_size_per_device
            and num_processes > 1
        ):
            raise Exception(
                "effective_batch_size equals batch_size_per_device, but num_proc more than 1"
            )

        assert self.effective_batch_size % num_processes == 0
        effective_batch_size_per_device = self.effective_batch_size // num_processes
        assert effective_batch_size_per_device % self.batch_size_per_device == 0
        self.gradient_accumulation_steps = (
            effective_batch_size_per_device // self.batch_size_per_device
        )

        print(
            f"acum={self.gradient_accumulation_steps} * device_bs={self.batch_size_per_device} * np={num_processes} = eff_bs{self.effective_batch_size}"
        )

        if self.eval_batch_size is None:
            self.eval_batch_size = self.batch_size_per_device * num_processes
