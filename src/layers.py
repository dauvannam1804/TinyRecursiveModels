import torch
import torch.nn as nn
import math
from typing import Optional

class SelfAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # x: [batch, seq_len, d_model]
        batch_size, seq_len, _ = x.size()
        
        # Projections
        # [batch, seq_len, d_model] -> [batch, seq_len, n_heads, d_head] -> [batch, n_heads, seq_len, d_head]
        q = self.w_q(x).view(batch_size, seq_len, self.n_heads, self.d_head).transpose(1, 2)
        k = self.w_k(x).view(batch_size, seq_len, self.n_heads, self.d_head).transpose(1, 2)
        v = self.w_v(x).view(batch_size, seq_len, self.n_heads, self.d_head).transpose(1, 2)
        
        # Scaled Dot-Product Attention
        # scores: [batch, n_heads, seq_len, seq_len]
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_head)
        
        if mask is not None:
            # mask: [seq_len, seq_len] or [batch, 1, seq_len, seq_len]
            # We assume mask has -inf for masked positions and 0 for valid ones
            if mask.dim() == 2:
                mask = mask.unsqueeze(0).unsqueeze(0)
            scores = scores + mask
            
        attn_probs = torch.softmax(scores, dim=-1)
        attn_probs = self.dropout(attn_probs)
        
        # Context
        # [batch, n_heads, seq_len, seq_len] @ [batch, n_heads, seq_len, d_head] -> [batch, n_heads, seq_len, d_head]
        context = torch.matmul(attn_probs, v)
        
        # Combine heads
        # [batch, n_heads, seq_len, d_head] -> [batch, seq_len, n_heads, d_head] -> [batch, seq_len, d_model]
        context = context.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
        
        # Output projection
        output = self.w_o(context)
        return output

class MLP(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1):
        super().__init__()
        self.fc1 = nn.Linear(d_model, 4 * d_model)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(4 * d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.act(x)
        x = self.dropout(x) # Dropout after activation usually? Or after projection? 
        # Standard Transformer: Linear -> GELU -> Linear -> Dropout
        # Let's follow standard:
        x = self.fc2(x)
        x = self.dropout(x)
        return x

class AddNorm(nn.Module):
    """
    Post-LN: LayerNorm(x + Sublayer(x))
    """
    def __init__(self, d_model: int, dropout: float = 0.1):
        super().__init__()
        self.ln = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x: torch.Tensor, sublayer_output: torch.Tensor) -> torch.Tensor:
        # We assume sublayer_output already has dropout applied if it's from MLP/Attn?
        # Standard: x + Dropout(Sublayer(x)) -> Norm
        # In my MLP/Attn above, I included dropout at the end.
        # So here we just add and norm.
        # But wait, usually the block handles the residual dropout.
        # Let's be safe: usually the sublayer returns the projected value, and the block adds dropout.
        # My MLP has dropout at the end. My SelfAttention has dropout on weights but not on output projection?
        # Standard implementation:
        # Attn: return linear(context)
        # Block: x = norm(x + dropout(attn(x)))
        
        # Let's adjust:
        # My SelfAttention returns w_o(context). No dropout on output.
        # My MLP returns dropout(fc2(...)).
        
        # Let's make AddNorm robust:
        # It takes x and sublayer_output.
        # It adds them and norms.
        # If sublayer_output needs dropout, it should be done before passing here or inside here.
        # Let's assume sublayer_output is "raw" and we apply dropout here?
        # Or we trust the sublayer.
        
        # To be consistent with "Add & Norm" diagram:
        # It usually means: Output = LayerNorm(Input + SublayerOutput)
        return self.ln(x + sublayer_output)
