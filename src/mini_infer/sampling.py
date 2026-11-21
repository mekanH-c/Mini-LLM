"""Token sampling algorithms: Greedy, Temperature, Top-K, and Top-P (Nucleus)."""

from typing import Optional
import torch
import torch.nn.functional as F


def sample_greedy(logits: torch.Tensor) -> torch.Tensor:
    """Greedy sampling: selects the token index with maximum logit.

    Args:
        logits: Unnormalized log probabilities of shape (batch_size, vocab_size)

    Returns:
        Tensor of shape (batch_size, 1) containing selected token indices.
    """
    return torch.argmax(logits, dim=-1, keepdim=True)


def sample_top_k(logits: torch.Tensor, top_k: int) -> torch.Tensor:
    """Filters logits keeping only the top-k highest scoring tokens.

    All logits outside the top-k are masked to -infinity.

    Args:
        logits: Tensor of shape (batch_size, vocab_size)
        top_k: Number of highest probability vocabulary tokens to keep (k > 0)

    Returns:
        Masked logits of shape (batch_size, vocab_size)
    """
    if top_k <= 0 or top_k >= logits.shape[-1]:
        return logits

    # Get the value of the k-th highest logit
    top_k_values, _ = torch.topk(logits, top_k, dim=-1)
    kth_value = top_k_values[..., -1, None]

    # Mask everything below the k-th logit to -inf
    mask = logits < kth_value
    return logits.masked_fill(mask, float("-inf"))


def sample_top_p(logits: torch.Tensor, top_p: float) -> torch.Tensor:
    """Nucleus (Top-p) filtering: keeps the smallest set of tokens whose cumulative probability >= top_p.

    Tokens outside the nucleus are masked to -infinity.

    Args:
        logits: Tensor of shape (batch_size, vocab_size)
        top_p: Cumulative probability threshold (0.0 < top_p <= 1.0)

    Returns:
        Masked logits of shape (batch_size, vocab_size)
    """
    if top_p >= 1.0:
        return logits

    # Sort logits descending
    sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
    sorted_probs = F.softmax(sorted_logits, dim=-1)

    # Compute cumulative probabilities
    cumulative_probs = torch.cumsum(sorted_probs, dim=-1)

    # Mask tokens with cumulative probability above threshold
    sorted_indices_to_remove = cumulative_probs > top_p

    # Shift right by 1 so the first token that exceeds top_p is still kept
    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
    sorted_indices_to_remove[..., 0] = False

    # Scatter mask back to original logit indexing
    indices_to_remove = sorted_indices_to_remove.scatter(
        dim=-1, index=sorted_indices, src=sorted_indices_to_remove
    )

    return logits.masked_fill(indices_to_remove, float("-inf"))


def sample(
    logits: torch.Tensor,
    temperature: float = 1.0,
    top_k: Optional[int] = None,
    top_p: Optional[float] = None,
) -> torch.Tensor:
    """Samples next token from logits given temperature, top-k, and top-p hyperparameters.

    Sampling Pipeline:
        1. Temperature == 0.0 -> Greedy argmax.
        2. Apply temperature scaling: logits = logits / temperature.
        3. Apply Top-K filtering.
        4. Apply Top-P (Nucleus) filtering.
        5. Softmax to probability distribution.
        6. Categorical multinomial draw.

    Args:
        logits: Unnormalized next-token logits of shape (batch_size, vocab_size)
        temperature: Scaling factor for distribution sharpness (0.0 = greedy)
        top_k: Number of highest probability tokens to keep (None or <= 0 disables)
        top_p: Nucleus probability mass threshold (None or >= 1.0 disables)

    Returns:
        Sampled token index tensor of shape (batch_size, 1)
    """
    # Deterministic greedy path
    if temperature == 0.0:
        return sample_greedy(logits)

    # Temperature scaling
    logits = logits / temperature

    # Top-K filtering
    if top_k is not None and top_k > 0:
        logits = sample_top_k(logits, top_k)

    # Top-P filtering
    if top_p is not None and 0.0 < top_p < 1.0:
        logits = sample_top_p(logits, top_p)

    # Compute probabilities in float32 for sampling stability
    probs = F.softmax(logits, dim=-1, dtype=torch.float32)

    # Categorical draw
    next_token = torch.multinomial(probs, num_samples=1)
    return next_token
