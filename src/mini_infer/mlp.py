"""SwiGLU Multi-Layer Perceptron (MLP) for LLaMA architectures."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from mini_infer.config import LlamaConfig


class SwiGLUMLP(nn.Module):
    """Feed-Forward Network with SwiGLU (Swish Gated Linear Unit) activation.

    Standard Transformer MLPs use:
        FFN(x) = Activation(x * W1) * W2

    SwiGLU introduces a gating mechanism that multiplies the activated gate branch
    element-wise with a linear up-projection branch:
        SwiGLU(x) = (SiLU(x * W_gate) * (x * W_up)) * W_down

    Args:
        config: LlamaConfig containing hidden_size and intermediate_size.
    """

    def __init__(self, config: LlamaConfig) -> None:
        super().__init__()
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size

        # Three projection matrices without bias
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass for SwiGLU MLP.

        Args:
            x: Input tensor of shape (batch_size, seq_len, hidden_size)

        Returns:
            Output tensor of shape (batch_size, seq_len, hidden_size)
        """
        # Shape transitions:
        # x: (batch_size, seq_len, hidden_size)
        # gate: (batch_size, seq_len, intermediate_size)
        # up: (batch_size, seq_len, intermediate_size)
        # activated: (batch_size, seq_len, intermediate_size)
        # down_proj: (batch_size, seq_len, hidden_size)
        gate = F.silu(self.gate_proj(x))
        up = self.up_proj(x)
        return self.down_proj(gate * up)
