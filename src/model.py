import torch
import torch.nn as nn
from dataclasses import dataclass
from typing import Optional, Tuple

# Re-import config to ensure type safety if needed, 
# but we can just use the dataclass definition or pass args.
from src.config import ModelConfig

class RecursiveBlock(nn.Module):
    """
    A standard Transformer block (Self-Attention + FFN) that will be reused recursively.
    """
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.ln1 = nn.LayerNorm(config.d_model)
        self.attn = nn.MultiheadAttention(
            embed_dim=config.d_model,
            num_heads=config.n_heads,
            dropout=config.dropout,
            batch_first=True
        )
        self.ln2 = nn.LayerNorm(config.d_model)
        self.ffn = nn.Sequential(
            nn.Linear(config.d_model, 4 * config.d_model),
            nn.GELU(),
            nn.Linear(4 * config.d_model, config.d_model),
            nn.Dropout(config.dropout)
        )
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # x: [batch, seq_len, d_model]
        # mask: [batch, seq_len] or [seq_len, seq_len] (attn_mask)
        
        # Self-Attention
        residual = x
        x = self.ln1(x)
        
        # nn.MultiheadAttention expects key_padding_mask as (batch, seq_len)
        # and attn_mask as (seq_len, seq_len) or (batch*num_heads, seq_len, seq_len)
        # Here we assume causal masking is handled by the caller or we pass it.
        # For simplicity, let's assume mask is the causal mask.
        
        attn_output, _ = self.attn(x, x, x, attn_mask=mask, need_weights=False)
        x = residual + self.dropout(attn_output)
        
        # FFN
        residual = x
        x = self.ln2(x)
        x = self.ffn(x)
        x = residual + self.dropout(x)
        
        return x

class TinyRecursiveModel(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        
        self.token_embedding = nn.Embedding(config.vocab_size, config.d_model)
        self.pos_embedding = nn.Embedding(config.max_seq_len, config.d_model)
        self.dropout = nn.Dropout(config.dropout)
        
        # The recursive block
        # We can stack a few layers inside one block if we want "depth" per step
        self.layers = nn.ModuleList([
            RecursiveBlock(config) for _ in range(config.n_layers)
        ])
        
        self.ln_f = nn.LayerNorm(config.d_model)
        self.head = nn.Linear(config.d_model, config.vocab_size, bias=False)

    def forward(self, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None):
        # input_ids: [batch, seq_len]
        batch_size, seq_len = input_ids.size()
        
        # Embeddings
        positions = torch.arange(seq_len, device=input_ids.device).unsqueeze(0)
        x = self.token_embedding(input_ids) + self.pos_embedding(positions)
        x = self.dropout(x)
        
        # Causal Mask
        # mask[i, j] = -inf if j > i else 0
        causal_mask = torch.triu(torch.ones(seq_len, seq_len, device=input_ids.device) * float('-inf'), diagonal=1)
        
        # Recursive Loop with Deep Supervision
        all_logits = []
        
        # Initial state H_0 is the embedding
        h = x
        
        for step in range(self.config.n_recurrence):
            # Pass through the block (which may have multiple layers)
            for layer in self.layers:
                h = layer(h, mask=causal_mask)
            
            # Calculate logits at this step
            # We normalize before the head
            logits = self.head(self.ln_f(h))
            all_logits.append(logits)
            
        # Return all logits for deep supervision
        # Shape: [n_recurrence, batch, seq_len, vocab_size]
        return torch.stack(all_logits)

if __name__ == "__main__":
    # Test model
    cfg = ModelConfig()
    model = TinyRecursiveModel(cfg)
    print("Model created.")
    
    dummy_input = torch.randint(0, cfg.vocab_size, (2, 10)) # batch=2, seq=10
    outputs = model(dummy_input)
    print("Output shape:", outputs.shape) 
    # Expected: [n_recurrence, batch, seq_len, vocab_size]
    assert outputs.shape == (cfg.n_recurrence, 2, 10, cfg.vocab_size)
    print("Test passed.")
