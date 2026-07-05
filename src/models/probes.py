"""
Probe architectures for representation analysis.

Two probes:
- LinearProbe: single linear layer. Tests if information is linearly accessible.
- MLPProbe: one hidden layer + ReLU. Tests if information is present (possibly nonlinear).

Both are intentionally simple. Complex probes risk learning the task rather than
reading information from the representation.
"""

import torch
import torch.nn as nn


class LinearProbe(nn.Module):
    """
    Linear probe: W @ x + b
    
    The strictest test of information encoding.
    If this succeeds, information is linearly separable in the representation space.
    
    Parameters: input_dim × num_classes + num_classes (bias)
    """
    
    def __init__(self, input_dim: int, num_classes: int):
        super().__init__()
        self.linear = nn.Linear(input_dim, num_classes)
        self.input_dim = input_dim
        self.num_classes = num_classes
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch_size, input_dim)
        Returns:
            logits: (batch_size, num_classes)
        """
        return self.linear(x)
    
    @property
    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())
    
    def __repr__(self) -> str:
        return (
            f"LinearProbe(input_dim={self.input_dim}, "
            f"num_classes={self.num_classes}, "
            f"params={self.num_parameters:,})"
        )


class MLPProbe(nn.Module):
    """
    One-hidden-layer MLP probe: Linear → ReLU → Dropout → Linear
    
    Less strict than linear. Captures nonlinearly encoded information.
    The gap between MLP and Linear accuracy tells us about encoding structure:
    - Small gap: information is linearly organized (clean encoding)
    - Large gap: information is present but entangled (needs nonlinear readout)
    
    Parameters: input_dim × hidden_dim + hidden_dim + hidden_dim × num_classes + num_classes
    """
    
    def __init__(
        self,
        input_dim: int,
        num_classes: int,
        hidden_dim: int = 256,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.num_classes = num_classes
        self.hidden_dim = hidden_dim
        
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch_size, input_dim)
        Returns:
            logits: (batch_size, num_classes)
        """
        return self.network(x)
    
    @property
    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())
    
    def __repr__(self) -> str:
        return (
            f"MLPProbe(input_dim={self.input_dim}, "
            f"hidden_dim={self.hidden_dim}, "
            f"num_classes={self.num_classes}, "
            f"params={self.num_parameters:,})"
        )


def create_probe(
    probe_type: str,
    input_dim: int,
    num_classes: int,
    hidden_dim: int = 256,
    dropout: float = 0.1,
) -> nn.Module:
    """
    Factory function to create a probe from config parameters.
    
    Args:
        probe_type: "linear" or "mlp"
        input_dim: Input feature dimension (model hidden_dim)
        num_classes: Number of output classes
        hidden_dim: MLP hidden layer size (ignored for linear)
        dropout: Dropout rate (ignored for linear)
    
    Returns:
        Probe module
    """
    if probe_type == "linear":
        return LinearProbe(input_dim, num_classes)
    elif probe_type == "mlp":
        return MLPProbe(input_dim, num_classes, hidden_dim, dropout)
    else:
        raise ValueError(f"Unknown probe type: {probe_type}. Use 'linear' or 'mlp'.")
