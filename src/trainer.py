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
        
        history = {"train_loss": [], "val_loss": []}
        
        for epoch in range(self.config.train.num_epochs):
            total_loss = 0
            progress_bar = tqdm(self.train_loader, desc=f"Epoch {epoch+1}/{self.config.train.num_epochs}")
            
            for batch in progress_bar:
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                labels = batch["labels"].to(self.device)
                
                self.optimizer.zero_grad()
                
                # Initialize y and z
                batch_size, seq_len = input_ids.size()
                y = torch.zeros(batch_size, seq_len, self.config.model.d_model, device=self.device)
                z = torch.zeros(batch_size, seq_len, self.config.model.d_model, device=self.device)
                
                batch_loss = 0
                
                # Deep Supervision Loop
                for step in range(self.config.model.n_supervision_steps):
                    y, z, logits = self.model(input_ids, attention_mask, y_init=y, z_init=z)
                    step_loss = self.criterion(logits.view(-1, self.config.model.vocab_size), labels.view(-1))
                    
                    weighted_loss = step_loss / self.config.model.n_supervision_steps
                    weighted_loss.backward()
                    
                    batch_loss += weighted_loss.item()
                    
                    y = y.detach()
                    z = z.detach()
                
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.train.grad_clip)
                self.optimizer.step()
                
                total_loss += batch_loss
                progress_bar.set_postfix({"loss": batch_loss})
                
            avg_loss = total_loss / len(self.train_loader)
            history["train_loss"].append(avg_loss)
            print(f"Epoch {epoch+1} Train Loss: {avg_loss:.4f}")
            
            # Validation
            if self.val_loader:
                val_loss = self.evaluate()
                history["val_loss"].append(val_loss)
                print(f"Epoch {epoch+1} Val Loss: {val_loss:.4f}")
            else:
                history["val_loss"].append(None)
            
            if self.config.train.output_dir:
                self.save_checkpoint(epoch, avg_loss)
                self.save_history(history)

    def evaluate(self):
        self.model.eval()
        total_loss = 0
        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc="Validating"):
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                labels = batch["labels"].to(self.device)
                
                batch_size, seq_len = input_ids.size()
                y = torch.zeros(batch_size, seq_len, self.config.model.d_model, device=self.device)
                z = torch.zeros(batch_size, seq_len, self.config.model.d_model, device=self.device)
                
                batch_loss = 0
                for step in range(self.config.model.n_supervision_steps):
                    y, z, logits = self.model(input_ids, attention_mask, y_init=y, z_init=z)
                    step_loss = self.criterion(logits.view(-1, self.config.model.vocab_size), labels.view(-1))
                    batch_loss += (step_loss.item() / self.config.model.n_supervision_steps)
                    
                total_loss += batch_loss
                
        self.model.train()
        return total_loss / len(self.val_loader)

    def save_history(self, history):
        import json
        os.makedirs(self.config.train.output_dir, exist_ok=True)
        path = os.path.join(self.config.train.output_dir, "history.json")
        with open(path, "w") as f:
            json.dump(history, f)

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

