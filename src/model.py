import torch
import torch.nn as nn
from typing import Optional, Tuple
from src.config import ModelConfig

class TRMCell(nn.Module):
    """
    The single tiny network used for both latent update and answer update.
    """
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=config.d_model,
                nhead=config.n_heads,
                dim_feedforward=4 * config.d_model,
                dropout=config.dropout,
                activation="gelu",
                batch_first=True,
                norm_first=True
            ) for _ in range(config.n_layers)
        ])
        
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x, src_mask=mask)
        return x

class TinyRecursiveModel(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        
        self.token_embedding = nn.Embedding(config.vocab_size, config.d_model)
        self.pos_embedding = nn.Embedding(config.max_seq_len, config.d_model)
        self.dropout = nn.Dropout(config.dropout)
        
        # The single tiny network
        self.net = TRMCell(config)
        
        self.ln_f = nn.LayerNorm(config.d_model)
        self.head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        
        # Q-head for Adaptive Computation Time (ACT)
        # Predicts halting probability (scalar per sequence or token? Paper says per example)
        # Usually pooled or max over sequence. Let's assume per-token for now or pool.
        # Pseudocode implies q_hat is compared to (y_hat == y_true), so it might be per token?
        # "loss += binary_cross_entropy(q_hat, (y_hat == y_true))" implies q_hat has same shape as y_hat (or compatible).
        # Let's assume q_hat is per token to predict if that token is correct.
        self.q_head = nn.Linear(config.d_model, 1) 

    def latent_recursion(self, x: torch.Tensor, y: torch.Tensor, z: torch.Tensor, mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        def latent_recursion(x, y, z, n=6):
            for i in range(n): # latent reasoning
                z = net(x, y, z)
            y = net(y, z) # refine output answer
            return y, z
        """
        # Latent reasoning (n times)
        for _ in range(self.config.n_latent_steps):
            # Input: x + y + z
            combined = x + y + z
            z = self.net(combined, mask=mask)
            
        # Refine output answer (1 time)
        # Input: y + z (x is NOT included)
        combined_y = y + z
        y = self.net(combined_y, mask=mask)
        
        return y, z

    def deep_recursion(self, x: torch.Tensor, y: torch.Tensor, z: torch.Tensor, mask: Optional[torch.Tensor] = None) -> Tuple[Tuple[torch.Tensor, torch.Tensor], torch.Tensor, torch.Tensor]:
        """
        def deep_recursion(x, y, z, n=6, T=3):
            # recursing T-1 times to improve y and z (no gradients needed)
            with torch.no_grad():
                for j in range(T-1):
                    y, z = latent_recursion(x, y, z, n)
            # recursing once to improve y and z
            y, z = latent_recursion(x, y, z, n)
            return (y.detach(), z.detach()), output_head(y), Q_head(y)
        """
        # Recursing T-1 times (no gradients)
        with torch.no_grad():
            for _ in range(self.config.n_recursion_steps - 1):
                y, z = self.latent_recursion(x, y, z, mask=mask)
                
        # Recursing once (with gradients)
        y, z = self.latent_recursion(x, y, z, mask=mask)
        
        # Outputs
        logits = self.head(self.ln_f(y))
        q_hat = torch.sigmoid(self.q_head(y)).squeeze(-1) # [batch, seq_len]
        
        # Return detached y, z for next supervision step, and current logits/q_hat
        return (y.detach(), z.detach()), logits, q_hat

    def forward(self, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None, 
                y_init: Optional[torch.Tensor] = None, z_init: Optional[torch.Tensor] = None) -> Tuple[Tuple[torch.Tensor, torch.Tensor], torch.Tensor, torch.Tensor]:
        """
        Wrapper to handle embedding and calling deep_recursion.
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
            
        return self.deep_recursion(x, y_init, z_init, mask=causal_mask)

if __name__ == "__main__":
    cfg = ModelConfig()
    model = TinyRecursiveModel(cfg)
    print("Model created.")
    
    dummy_input = torch.randint(0, cfg.vocab_size, (2, 10))
    (y_next, z_next), logits, q_hat = model(dummy_input)
    print("Logits shape:", logits.shape)
    print("Q_hat shape:", q_hat.shape)
    assert logits.shape == (2, 10, cfg.vocab_size)
    assert q_hat.shape == (2, 10)
    print("Test passed.")
