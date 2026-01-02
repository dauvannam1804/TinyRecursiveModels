from dataclasses import dataclass, field
from typing import Optional

@dataclass
class ModelConfig:
    d_model: int = 256
    n_heads: int = 4
    n_layers: int = 2  # Layers in the tiny network (4x in diagram)
    # TRM Specifics
    n_latent_steps: int = 1 #2 #3 #6  # n in paper (z update loop)
    n_recursion_steps: int = 1 #2 #3 #3 # T in paper (Deep Recursion loop)
    n_supervision_steps: int = 1 #4 #8 #16 # N_sup in paper
    vocab_size: int = 50000 # Will be updated after tokenizer training
    max_seq_len: int = 512
    dropout: float = 0.1

@dataclass
class TrainConfig:
    batch_size: int = 4
    learning_rate: float = 3e-4
    weight_decay: float = 0.01
    num_epochs: int = 5
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
    # data_path: str = "data/processed/xlam_1k_swift.json"
    # val_data_path: str = "data/processed/xlam_val_200_swift.json"
    data_path: str = "data/processed/xlam_train_20k_swift.json"
    val_data_path: str = "data/processed/xlam_val_2k_swift.json"
    tokenizer_path: str = "data/processed/tokenizer.json"
