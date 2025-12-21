import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Optional
import os
from tqdm import tqdm
import wandb
from src.config import Config

class Trainer:
    def __init__(self, model: nn.Module, train_loader: DataLoader, config: Config, val_loader: Optional[DataLoader] = None):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(), 
            lr=config.train.learning_rate, 
            weight_decay=config.train.weight_decay
        )
        
        self.criterion = nn.CrossEntropyLoss(ignore_index=-100)
        
    def train(self):
        print(f"Starting training on {self.device}...")
        self.model.train()
        
        for epoch in range(self.config.train.num_epochs):
            total_loss = 0
            progress_bar = tqdm(self.train_loader, desc=f"Epoch {epoch+1}/{self.config.train.num_epochs}")
            
            for batch in progress_bar:
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                labels = batch["labels"].to(self.device)
                
                self.optimizer.zero_grad()
                
                # Forward pass
                # outputs shape: [n_recurrence, batch, seq_len, vocab_size]
                outputs = self.model(input_ids, attention_mask)
                
                # Deep Supervision Loss
                loss = 0
                n_steps = outputs.shape[0]
                
                # We can weight the loss at each step differently if we want
                # For now, uniform weighting or just sum
                for step in range(n_steps):
                    step_logits = outputs[step]
                    # Flatten for CrossEntropyLoss
                    step_loss = self.criterion(step_logits.view(-1, self.config.model.vocab_size), labels.view(-1))
                    loss += step_loss
                
                # Normalize loss by number of steps if needed, or keep as sum
                loss = loss / n_steps
                
                loss.backward()
                
                # Gradient Clipping
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.train.grad_clip)
                
                self.optimizer.step()
                
                total_loss += loss.item()
                progress_bar.set_postfix({"loss": loss.item()})
                
                # TODO: Add wandb logging here
                
            avg_loss = total_loss / len(self.train_loader)
            print(f"Epoch {epoch+1} completed. Avg Loss: {avg_loss:.4f}")
            
            if self.config.train.output_dir:
                self.save_checkpoint(epoch, avg_loss)

    def save_checkpoint(self, epoch, loss):
        os.makedirs(self.config.train.output_dir, exist_ok=True)
        path = os.path.join(self.config.train.output_dir, f"checkpoint_epoch_{epoch}.pt")
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'loss': loss,
        }, path)
        print(f"Checkpoint saved to {path}")

