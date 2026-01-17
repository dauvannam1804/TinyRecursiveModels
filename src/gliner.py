import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, List, Tuple

def create_projection_layer(hidden_size: int, dropout: float, out_dim: Optional[int] = None) -> nn.Sequential:
    """Creates a two-layer projection network with ReLU activation and dropout.
    """
    if out_dim is None:
        out_dim = hidden_size

    return nn.Sequential(
        nn.Linear(hidden_size, out_dim * 4), nn.ReLU(), nn.Dropout(dropout), nn.Linear(out_dim * 4, out_dim)
    )

def extract_elements(sequence, indices):
    """Extract elements from a sequence using provided indices.
    Args:
        sequence (torch.Tensor): Input sequence of shape [B, L, D].
        indices (torch.Tensor): Indices to extract, shape [B, K].
    Returns:
        torch.Tensor: Extracted elements of shape [B, K, D].
    """
    D = sequence.size(-1)
    # Expand indices to [B, K, D]
    expanded_indices = indices.unsqueeze(2).expand(-1, -1, D)
    # Gather the elements
    extracted_elements = torch.gather(sequence, 1, expanded_indices)
    return extracted_elements

class SpanMarkerV0(nn.Module):
    """Marks and projects span endpoints using an MLP.
    A cleaner version of SpanMarker using the create_projection_layer utility.
    """
    def __init__(self, hidden_size: int, max_width: int, dropout: float = 0.4):
        super().__init__()
        self.max_width = max_width
        self.project_start = create_projection_layer(hidden_size, dropout)
        self.project_end = create_projection_layer(hidden_size, dropout)
        self.out_project = create_projection_layer(hidden_size * 2, dropout, hidden_size)

    def forward(self, h: torch.Tensor, span_idx: torch.Tensor) -> torch.Tensor:
        B, L, D = h.size()
        start_rep = self.project_start(h)
        end_rep = self.project_end(h)
        start_span_rep = extract_elements(start_rep, span_idx[:, :, 0])
        end_span_rep = extract_elements(end_rep, span_idx[:, :, 1])
        cat = torch.cat([start_span_rep, end_span_rep], dim=-1).relu()
        return self.out_project(cat)

class SpanRepLayer(nn.Module):
    """Factory class for span representation, currently simplified for markerV0.
    """
    def __init__(self, hidden_size, max_width, span_mode, **kwargs):
        super().__init__()
        if span_mode == "markerV0":
            self.span_rep_layer = SpanMarkerV0(hidden_size, max_width, **kwargs)
        else:
            raise ValueError(f"Unknown or unsupported span mode {span_mode} for this simplified implementation.")

    def forward(self, x, *args):
        return self.span_rep_layer(x, *args)
