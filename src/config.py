from dataclasses import dataclass, field
from typing import Optional

@dataclass
class ModelConfig:
    d_model: int = 256
    n_heads: int = 4
    n_layers: int = 4  # Layers in the tiny network (4x in diagram)
    # TRM Specifics
    n_latent_steps: int = 6  # n in paper
    n_recursion_steps: int = 3 # T in paper (not used in current diagram logic but good to keep?)
    # Diagram logic uses n_latent_steps for z update loop.
    # It doesn't explicitly show T loops of (z_loop -> y_update).
    # But let's keep it if we want to support the full paper logic later, 
    # or just n_latent_steps is enough for the current implementation.
    # My current model implementation uses n_latent_steps.
    n_supervision_steps: int = 16 # N_sup in paper
    vocab_size: int = 32000 # Will be updated after tokenizer training
    max_seq_len: int = 256
    dropout: float = 0.1

@dataclass
class TrainConfig:
    batch_size: int = 4
    learning_rate: float = 3e-4
    weight_decay: float = 0.01
    num_epochs: int = 10
    warmup_steps: int = 100
    grad_clip: float = 1.0
    deep_supervision_weight: float = 1.0 # Weight for intermediate losses
    log_interval: int = 10
    save_interval: int = 1000
    output_dir: str = "checkpoints"
    project_name: str = "tiny-recursive-model"
    run_name: str = "trm-run-001"
    seed: int = 42

@dataclass
class Config:
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    data_path: str = "data/processed/sample_1k.csv"
    tokenizer_path: str = "data/processed/tokenizer.json"
