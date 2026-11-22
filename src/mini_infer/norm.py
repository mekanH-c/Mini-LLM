"""Root Mean Square Layer Normalization (RMSNorm)."""

import torch
import torch.nn as nn


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization (RMSNorm).

    Unlike standard LayerNorm, RMSNorm does not re-center activations by subtracting
    the mean, reducing compute while achieving comparable training stability.

    Formula:
        RMS(x) = sqrt(mean(x^2) + eps)
        y = (x / RMS(x)) * gamma

    Args:
        dim: Hidden dimension size of the input tensor.
        eps: Small constant added to the variance estimate to avoid division by zero.
    """

    def __init__(self, dim: int, eps: float = 1e-5) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x: torch.Tensor) -> torch.Tensor:
        """Compute the RMS-normalized tensor in float32 for numerical stability."""
        # Shape: x is (..., dim) -> mean over last dimension produces (..., 1)
        variance = x.pow(2).mean(dim=-1, keepdim=True)
        return x * torch.rsqrt(variance + self.eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass for RMSNorm.

        Args:
            x: Input tensor of shape (batch_size, seq_len, dim)

        Returns:
            Normalized and scaled tensor of shape (batch_size, seq_len, dim)
        """
        # Intermediate computation in float32 to prevent underflow/overflow
        normalized = self._norm(x.float()).type_as(x)
        return normalized * self.weight
