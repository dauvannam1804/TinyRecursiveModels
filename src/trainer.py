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
        self.bce_criterion = nn.BCELoss() # For Q-head
        
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
                
                # Initialize y and z (Let model handle it with learnable params)
                y, z = None, None
                
                batch_loss = 0
                
                # Deep Supervision Loop
                for step in range(self.config.model.n_supervision_steps):
                    # Forward pass
                    y, z, logits, q_hat, param_logits = self.model(input_ids, attention_mask, y_init=y, z_init=z)
                    
                    # Calculate Loss (Cross Entropy)
                    main_loss = self.criterion(logits.view(-1, self.config.model.vocab_size), labels.view(-1))
                    
                    # Param Loss (GLINEAR)
                    # We use the same labels for now, assuming the dataset provides token-level labels for params
                    # or we might need a separate label field if the task differs.
                    # For now, let's assume 'labels' contains the target tokens for both main and param tasks
                    # (or we use a placeholder/mask if not available).
                    param_loss = self.criterion(param_logits.view(-1, self.config.model.vocab_size), labels.view(-1))
                    
                    loss = main_loss + param_loss
                    
                    # ACT Loss (Q-head)
                    # Target: 1 if prediction is correct, 0 otherwise
                    with torch.no_grad():
                        preds = torch.argmax(logits, dim=-1)
                        valid_mask = labels != -100
                        correct = (preds == labels) & valid_mask
                        target_halt = correct.float()
                    
                    if valid_mask.any():
                        # q_hat is [batch, seq_len]
                        act_loss = self.bce_criterion(q_hat[valid_mask], target_halt[valid_mask])
                        loss += act_loss
                    
                    # Backward & Step (Step-wise Optimization)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.train.grad_clip)
                    self.optimizer.step()
                    self.optimizer.zero_grad()
                    
                    batch_loss += loss.item()
                    
                    # Detach for next step
                    y = y.detach()
                    z = z.detach()
                
                total_loss += batch_loss / self.config.model.n_supervision_steps
                progress_bar.set_postfix({"loss": batch_loss / self.config.model.n_supervision_steps})
                
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
                y, z = None, None # Use learnable init
                
                batch_loss = 0
                for step in range(self.config.model.n_supervision_steps):
                    y, z, logits, q_hat, param_logits = self.model(input_ids, attention_mask, y_init=y, z_init=z)
                    loss = self.criterion(logits.view(-1, self.config.model.vocab_size), labels.view(-1))
                    # Add param loss to validation metric too
                    param_loss = self.criterion(param_logits.view(-1, self.config.model.vocab_size), labels.view(-1))
                    loss += param_loss
                    batch_loss += loss.item()
                    
                total_loss += batch_loss / self.config.model.n_supervision_steps
                
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
