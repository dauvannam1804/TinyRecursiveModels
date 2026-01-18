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
                # print("input ids:", input_ids[0])
                # print("input ids shape:", input_ids[0].shape)
                attention_mask = batch["attention_mask"].to(self.device)
                # print("attention mask:", attention_mask[0])
                # print("attention mask shape:", attention_mask[0].shape)
                labels = batch["labels"].to(self.device)
                # print("labels:", labels[0])
                # print("labels shape:", labels[0].shape)
                
                # Initialize y and z (Let model handle it with learnable params)
                y, z = None, None
                
                batch_loss = 0
                
                # Fetch GLiNER inputs from batch if available
                span_idx = batch.get("span_idx").to(self.device) if "span_idx" in batch else None
                # print("span_idx:", span_idx)
                span_labels = batch.get("span_labels").to(self.device) if "span_labels" in batch else None
                # print("span_labels:", span_labels)
                prompts_ids = batch.get("prompts_ids").to(self.device) if "prompts_ids" in batch else None
                # print("prompts_ids:", prompts_ids)

                # Deep Supervision Loop
                for step in range(self.config.model.n_supervision_steps):
                    
                    prompts_embedding = None
                    if prompts_ids is not None:
                         # prompts_ids: [Batch, NumClasses, Length]
                         B, C, L = prompts_ids.size()
                         flat_ids = prompts_ids.view(B * C, L)
                         embeds = self.model.token_embedding(flat_ids)
                         # Mask padding (0)
                         mask = (flat_ids != 0).float().unsqueeze(-1)
                         sum_embeds = (embeds * mask).sum(dim=1)
                         count = mask.sum(dim=1).clamp(min=1)
                         avg_embeds = sum_embeds / count
                         prompts_embedding = avg_embeds.view(B, C, -1)

                    # Forward pass
                    y, z, logits, q_hat, span_scores = self.model(
                        input_ids, 
                        attention_mask, 
                        span_idx=span_idx,
                        prompts_embedding=prompts_embedding,
                        y_init=y, 
                        z_init=z
                    )
                    
                    # Calculate Loss (Cross Entropy) for Generation
                    main_loss = self.criterion(logits.view(-1, self.config.model.vocab_size), labels.view(-1))
                    
                    loss = main_loss
                    
                    # Span Classification Loss (GLiNER)
                    if span_scores is not None and span_labels is not None:
                        # span_scores: [B, NumSpans, NumClasses]
                        # span_labels: [B, NumSpans] (Class Indices)
                        # We use CE Loss here.
                        span_loss = self.criterion(span_scores.view(-1, self.config.model.num_classes), span_labels.view(-1))
                        loss += span_loss
                    
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

                    # print("END")
                    # exit(0)
                
                # Update total loss correctly (accumulate average batch loss)
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
                # Fetch GLiNER inputs from batch if available
                span_idx = batch.get("span_idx").to(self.device) if "span_idx" in batch else None
                span_labels = batch.get("span_labels").to(self.device) if "span_labels" in batch else None
                prompts_ids = batch.get("prompts_ids").to(self.device) if "prompts_ids" in batch else None

                prompts_embedding = None
                if prompts_ids is not None:
                     B, C, L = prompts_ids.size()
                     flat_ids = prompts_ids.view(B * C, L)
                     embeds = self.model.token_embedding(flat_ids)
                     mask = (flat_ids != 0).float().unsqueeze(-1)
                     sum_embeds = (embeds * mask).sum(dim=1)
                     count = mask.sum(dim=1).clamp(min=1)
                     avg_embeds = sum_embeds / count
                     prompts_embedding = avg_embeds.view(B, C, -1)
                
                for step in range(self.config.model.n_supervision_steps):
                    y, z, logits, q_hat, span_scores = self.model(
                        input_ids, 
                        attention_mask, 
                        span_idx=span_idx,
                        prompts_embedding=prompts_embedding,
                        y_init=y, 
                        z_init=z
                    )
                    loss = self.criterion(logits.view(-1, self.config.model.vocab_size), labels.view(-1))
                    
                    # Add span loss to validation metric too
                    if span_scores is not None and span_labels is not None:
                         span_loss = self.criterion(span_scores.view(-1, self.config.model.num_classes), span_labels.view(-1))
                         loss += span_loss
                    
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
