import torch
from torch.utils.data import DataLoader
from src.config import Config
from src.dataset import MathDataset
from src.model import TinyRecursiveModel
from src.trainer import Trainer
import os

def main():
    # Load Config
    config = Config()
    
    # Override config for quick test if needed
    # config.train.num_epochs = 1
    
    print("Initializing...")
    
    # Dataset & Dataloader
    # Ensure tokenizer exists
    if not os.path.exists(config.tokenizer_path):
        raise FileNotFoundError(f"Tokenizer not found at {config.tokenizer_path}. Please run 'scripts/train_tokenizer.sh' first.")
        
    train_dataset = MathDataset(config.data_path, config.tokenizer_path, config.model.max_seq_len)
    train_loader = DataLoader(train_dataset, batch_size=config.train.batch_size, shuffle=True)
    
    print(f"Dataset loaded. Size: {len(train_dataset)}")
    
    val_loader = None
    if config.val_data_path and os.path.exists(config.val_data_path):
        print(f"Loading validation data from {config.val_data_path}...")
        val_dataset = MathDataset(config.val_data_path, config.tokenizer_path, config.model.max_seq_len)
        val_loader = DataLoader(val_dataset, batch_size=config.train.batch_size, shuffle=False)
    
    # Model
    model = TinyRecursiveModel(config.model)
    print("Model initialized.")
    
    # Trainer
    trainer = Trainer(model, train_loader, config, val_loader=val_loader)
    
    # Train
    trainer.train()

if __name__ == "__main__":
    main()
