import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple

class MultiheadAttention(nn.Module):
    """
    Standard Multi-head attention (non-causal for Bi-Encoder).
    """
    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        batch_size, seq_len, _ = x.size()
        
        q = self.q_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Scaled Dot-Product Attention
        scores = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim ** 0.5)
        
        if mask is not None:
            scores = scores + mask
            
        attn_probs = torch.softmax(scores, dim=-1)
        attn_probs = self.dropout(attn_probs)
        
        context = torch.matmul(attn_probs, v)
        context = context.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
        
        return self.out_proj(context)

class SelfAttentionBlock(nn.Module):
    """
    Bi-Encoder Block: Pre-Norm -> Self-Attn -> Add -> Pre-Norm -> MLP -> Add
    """
    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = MultiheadAttention(d_model, num_heads, dropout)
        self.dropout1 = nn.Dropout(dropout)
        
        self.norm2 = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(4 * d_model, d_model),
            nn.Dropout(dropout)
        )
        
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # Pre-Norm architecture (standard for modern Transformers)
        x_norm = self.norm1(x)
        attn_out = self.attn(x_norm, mask)
        x = x + self.dropout1(attn_out)
        
        x_norm = self.norm2(x)
        mlp_out = self.mlp(x_norm)
        x = x + mlp_out
        return x

class SpanMarker(nn.Module):
    """
    Span Representation Layer:
    Concatenates [Start_Proj, End_Proj, Avg_Token_Proj] -> MLP
    """
    def __init__(self, d_model: int, max_width: int, dropout: float = 0.1):
        super().__init__()
        self.max_width = max_width
        self.d_model = d_model
        
        # Projections for start, end, and average token
        self.project_start = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.Dropout(dropout)
        )
        self.project_end = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.Dropout(dropout)
        )
        # Average token projection is implicit or can be added if needed. 
        # GLiNER uses raw average then concatenates.
        
        # Final projection: 3 * d_model -> d_model
        self.out_project = nn.Sequential(
            nn.Linear(d_model * 3, d_model),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model)
        )
        
    def forward(self, h: torch.Tensor) -> torch.Tensor:
        # h: [batch, seq_len, d_model]
        
        # We want to capture span-like information for each token.
        # Since we don't have explicit span indices, we will use a "sliding window" 
        # approach to simulate spans centered at each token.
        # We can use a Convolution to aggregate local context.
        
        # 1. Start/End Projections (Token-level)
        start_rep = self.project_start(h) # [B, L, D]
        end_rep = self.project_end(h)     # [B, L, D]
        
        # 2. Local Context (Simulating "Average Token" in a span)
        # We use a Conv1d to aggregate neighbors. Kernel size 3 means context of +/- 1 token.
        # We can use multiple kernel sizes to simulate different span widths if we want,
        # but let's start with a simple aggregation.
        
        # Transpose for Conv1d: [B, D, L]
        h_t = h.transpose(1, 2)
        
        # We'll use a simple average pooling or conv to get "span content"
        # Let's use a Conv1d with kernel size 3, padding 1 to keep length same.
        # This represents a span of length 3 centered at the token.
        # To be more robust, we could use multiple widths, but let's stick to one for now.
        # Actually, GLiNER uses [Start, End, Avg].
        # Let's approximate "Avg" with the token itself + neighbors.
        
        # For this implementation, we will just concatenate Start, End, and Original (as Avg).
        # But Start and End should probably be "shifted" to represent a span?
        # Without explicit spans, "Start" and "End" are just different views of the token.
        
        # Let's try to capture "Previous" and "Next" as Start/End?
        # Start = h[t-1], End = h[t+1]?
        # This would represent a span of 3 tokens centered at t.
        
        batch_size, seq_len, d_model = h.size()
        
        # Shifted features
        # Start (Previous token): [0, h[0], ..., h[n-1]]
        start_feat = torch.cat([torch.zeros(batch_size, 1, d_model, device=h.device), h[:, :-1, :]], dim=1)
        # End (Next token): [h[1], ..., h[n], 0]
        end_feat = torch.cat([h[:, 1:, :], torch.zeros(batch_size, 1, d_model, device=h.device)], dim=1)
        
        # Project
        start_proj = self.project_start(start_feat)
        end_proj = self.project_end(end_feat)
        
        # Concatenate: [Start_Proj, End_Proj, Center_Token]
        # This creates a representation for the "span" (t-1, t, t+1)
        span_rep = torch.cat([start_proj, end_proj, h], dim=-1)
        
        # Final Projection
        return self.out_project(span_rep)

class GLINEAR(nn.Module):
    """
    GLiNER-based Sub-network.
    """
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        # Bi-Directional Encoder (Stack of SelfAttentionBlocks)
        # We use 2 layers for the sub-network
        self.encoder = nn.ModuleList([
            SelfAttentionBlock(d_model, n_heads, dropout) for _ in range(2)
        ])
        
        # SpanMarker (Using sliding window context)
        self.span_marker = SpanMarker(d_model, max_width=12, dropout=dropout)
        
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # x: [batch, seq_len, d_model]
        
        # 1. Bi-Directional Encoding (or Causal if mask is provided)
        for layer in self.encoder:
            x = layer(x, mask=mask)
            
        # 2. Span Representation
        # Use SpanMarker to enhance token features with local context
        x = self.span_marker(x)
        
        return x
