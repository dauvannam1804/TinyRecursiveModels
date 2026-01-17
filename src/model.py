import torch
import torch.nn as nn
from typing import Optional, Tuple
from src.config import ModelConfig
from src.layers import SelfAttention, MLP, AddNorm
from src.gliner import SpanRepLayer, create_projection_layer

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
        
        # Q-head for Adaptive Computation Time (ACT)
        self.q_head = nn.Linear(config.d_model, 1)
        
        # GLiNER Span Representation
        self.span_rep_layer = SpanRepLayer(
            hidden_size=config.d_model,
            max_width=config.max_width,
            span_mode=config.span_mode,
            dropout=config.dropout
        )
        
        # Projection for Class/Prompt Embeddings (to match span_rep dimension)
        self.prompt_rep_layer = create_projection_layer(config.d_model, config.dropout)
        
        # Learnable Initialization for y and z
        self.y_init_param = nn.Parameter(torch.zeros(1, 1, config.d_model))
        self.z_init_param = nn.Parameter(torch.zeros(1, 1, config.d_model))
        # Optional: Initialize with small random values instead of pure zeros?
        nn.init.normal_(self.y_init_param, std=0.02)
        nn.init.normal_(self.z_init_param, std=0.02)

    def latent_recursion(self, x: torch.Tensor, y: torch.Tensor, z: torch.Tensor, mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Diagram Logic (matches 'latent_recursion' in pseudocode):
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
                span_idx: Optional[torch.Tensor] = None, prompts_embedding: Optional[torch.Tensor] = None,
                y_init: Optional[torch.Tensor] = None, z_init: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass for ONE supervision step.
        Returns: y_next, z_next, logits, q_hat, span_scores
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
            # Broadcast learnable parameter to [batch, seq_len, d_model]
            y_init = self.y_init_param.expand(batch_size, seq_len, -1)
        if z_init is None:
            z_init = self.z_init_param.expand(batch_size, seq_len, -1)
            
        y, z = y_init, z_init
        
        # Deep Recursion Loop (T times)
        # T-1 times without gradient
        with torch.no_grad():
            for _ in range(self.config.n_recursion_steps - 1):
                y, z = self.latent_recursion(x, y, z, mask=causal_mask)
        
        # Last step WITH gradient
        y, z = self.latent_recursion(x, y, z, mask=causal_mask)
        
        # Logits (Reverse Embedding)
        logits = self.head(self.ln_f(y))
        
        # Q-head (Halting Probability)
        q_hat = torch.sigmoid(self.q_head(y)).squeeze(-1) # [batch, seq_len]
        
        # GLiNER Span Scoring
        span_scores = None
        if span_idx is not None and prompts_embedding is not None:
             # y is [Batch, SeqLen, Hidden] corresponding to 'words_embedding' in GLiNER
             # span_idx is [Batch, NumSpans, 2]
             
             # 1. Compute Span Representations
             # Note: SpanRepLayer expects x to be [Batch, SeqLen, Hidden].
             # We use 'y' as the contextualized representation.
             span_rep = self.span_rep_layer(y, span_idx)  # [Batch, SeqLen, MaxWidth, Hidden]
             
             # 2. Project Prompts
             # prompts_embedding: [Batch, NumClasses, Hidden]
             prompts_proj = self.prompt_rep_layer(prompts_embedding) # [Batch, NumClasses, Hidden]
             
             # 3. Compute Scores (Dot Product)
             # span_rep: [B, NumSpans, D], prompts_proj: [B, NumClasses, D]
             # Output: [B, NumSpans, NumClasses]
             span_scores = torch.einsum("BSD,BCD->BSC", span_rep, prompts_proj)
        
        return y, z, logits, q_hat, span_scores

if __name__ == "__main__":
    cfg = ModelConfig()
    model = TinyRecursiveModel(cfg)
    import torch
    print("Model created.")
    
    batch_size = 2
    seq_len = 10
    num_classes = cfg.num_classes
    max_width = cfg.max_width
    
    dummy_input = torch.randint(0, cfg.vocab_size, (batch_size, seq_len))
    
    # Create dummy span_idx: [Batch, NumSpans, 2]
    # Let's say we have 5 spans per sequence
    num_spans = 5
    span_idx = torch.randint(0, seq_len, (batch_size, num_spans, 2))
    
    # Create dummy prompts_embedding: [Batch, NumClasses, Hidden]
    prompts_embedding = torch.randn(batch_size, num_classes, cfg.d_model)
    
    y, z, logits, q_hat, span_scores = model(dummy_input, span_idx=span_idx, prompts_embedding=prompts_embedding)
    
    print("Logits shape:", logits.shape)
    print("Q_hat shape:", q_hat.shape)
    if span_scores is not None:
        print("Span Scores shape:", span_scores.shape)
        assert span_scores.shape == (batch_size, num_spans, num_classes)
        print("Test passed.")
    else:
        print("Span scores is None.")
