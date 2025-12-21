import torch
import torch.nn as nn
from typing import Optional, Tuple
from src.config import ModelConfig
from src.layers import SelfAttention, MLP, AddNorm

class TRMBlock(nn.Module):
    """
    Single Transformer Block:
    Self-Attention -> Add & Norm -> MLP -> Add & Norm
    """
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.attn = SelfAttention(config.d_model, config.n_heads, config.dropout)
        self.add_norm1 = AddNorm(config.d_model, config.dropout)
        
        self.mlp = MLP(config.d_model, config.dropout)
        self.add_norm2 = AddNorm(config.d_model, config.dropout)
        
        # We need an extra dropout for the attention output before residual?
        # My AddNorm implementation assumes sublayer_output is ready to add.
        # In src/layers.py, SelfAttention does NOT have output dropout.
        # MLP DOES have output dropout.
        # So we should add dropout for attn here.
        self.dropout = nn.Dropout(config.dropout)
        
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # 1. Self-Attention -> Add & Norm
        attn_out = self.attn(x, mask)
        attn_out = self.dropout(attn_out)
        x = self.add_norm1(x, attn_out)
        
        # 2. MLP -> Add & Norm
        mlp_out = self.mlp(x)
        # MLP in layers.py already has dropout at the end
        x = self.add_norm2(x, mlp_out)
        
        return x

class TRMCell(nn.Module):
    """
    The single tiny network used for both latent update and answer update.
    Diagram: 4x (4 layers)
    """
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.layers = nn.ModuleList([
            TRMBlock(config) for _ in range(config.n_layers)
        ])
        
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x, mask)
        return x

class TinyRecursiveModel(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        
        self.token_embedding = nn.Embedding(config.vocab_size, config.d_model)
        self.pos_embedding = nn.Embedding(config.max_seq_len, config.d_model)
        self.dropout = nn.Dropout(config.dropout)
        
        # The single tiny network (4 layers)
        self.net = TRMCell(config)
        
        self.ln_f = nn.LayerNorm(config.d_model)
        self.head = nn.Linear(config.d_model, config.vocab_size, bias=False)

    def forward_step(self, x: torch.Tensor, y: torch.Tensor, z: torch.Tensor, mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Diagram Logic:
        1. Update z (n times): Input = x + y + z. Output -> new z.
        2. Update y (1 time): Input = y + z. Output -> new y.
        """
        # 1. Update z (n times)
        for _ in range(self.config.n_latent_steps):
            # Input: x + y + z
            combined = x + y + z
            z = self.net(combined, mask=mask)
            
        # 2. Update y (1 time)
        # Input: y + z (x is NOT included)
        combined_y = y + z
        y = self.net(combined_y, mask=mask)
        
        return y, z

    def forward(self, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None, 
                y_init: Optional[torch.Tensor] = None, z_init: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass for ONE supervision step.
        Returns: y_next, z_next, logits
        """
        batch_size, seq_len = input_ids.size()
        device = input_ids.device
        
        # Embeddings
        positions = torch.arange(seq_len, device=device).unsqueeze(0)
        x = self.token_embedding(input_ids) + self.pos_embedding(positions)
        x = self.dropout(x)
        
        # Causal Mask
        causal_mask = torch.triu(torch.ones(seq_len, seq_len, device=device) * float('-inf'), diagonal=1)
        
        # Initialize y and z if not provided
        if y_init is None:
            y_init = torch.zeros_like(x)
        if z_init is None:
            z_init = torch.zeros_like(x)
            
        y, z = y_init, z_init
        
        # Update step
        y, z = self.forward_step(x, y, z, mask=causal_mask)
        
        # Logits (Reverse Embedding)
        logits = self.head(self.ln_f(y))
        
        return y, z, logits

if __name__ == "__main__":
    cfg = ModelConfig()
    model = TinyRecursiveModel(cfg)
    print("Model created.")
    
    dummy_input = torch.randint(0, cfg.vocab_size, (2, 10))
    y, z, logits = model(dummy_input)
    print("Logits shape:", logits.shape)
    assert logits.shape == (2, 10, cfg.vocab_size)
    print("Test passed.")
