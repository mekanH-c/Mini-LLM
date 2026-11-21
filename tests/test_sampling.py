"""Unit tests for token sampling strategies (greedy, temperature, top-k, top-p)."""

import torch
import pytest
from mini_infer.sampling import sample_greedy, sample_top_k, sample_top_p, sample


def test_sample_greedy():
    # Batch size 2, Vocab size 5
    logits = torch.tensor([
        [1.0, 2.5, 0.2, 5.1, 3.0],   # max is index 3 (5.1)
        [10.0, 2.0, 3.0, 1.0, 0.5],  # max is index 0 (10.0)
    ])
    selected = sample_greedy(logits)
    assert selected.shape == (2, 1)
    assert selected[0, 0].item() == 3
    assert selected[1, 0].item() == 0


def test_sample_temperature_zero():
    logits = torch.tensor([[1.0, 4.0, 2.0]])
    selected = sample(logits, temperature=0.0)
    assert selected.item() == 1


def test_sample_top_k():
    logits = torch.tensor([[1.0, 5.0, 3.0, 2.0, 4.0]])
    # Top 2 should keep only indices 1 (5.0) and 4 (4.0)
    filtered = sample_top_k(logits, top_k=2)
    assert filtered[0, 1] == 5.0
    assert filtered[0, 4] == 4.0
    assert filtered[0, 0] == float("-inf")
    assert filtered[0, 2] == float("-inf")
    assert filtered[0, 3] == float("-inf")


def test_sample_top_p():
    # Probs: [0.5, 0.3, 0.1, 0.1]
    # Logits proportional to log probs
    probs = torch.tensor([[0.5, 0.3, 0.1, 0.1]])
    logits = torch.log(probs)

    # top_p = 0.75: sorted cumulative: 0.5 (keep), 0.8 (crosses 0.75, keep first token that exceeds), 0.9 (mask), 1.0 (mask)
    # So indices 0 and 1 should remain valid
    filtered = sample_top_p(logits, top_p=0.75)
    assert not torch.isinf(filtered[0, 0])
    assert not torch.isinf(filtered[0, 1])
    assert torch.isinf(filtered[0, 2])
    assert torch.isinf(filtered[0, 3])


def test_sample_stochastic_distribution():
    torch.manual_seed(42)
    # Huge logit at index 2, small elsewhere
    logits = torch.tensor([[0.0, 0.0, 100.0, 0.0]])
    draws = [sample(logits, temperature=1.0).item() for _ in range(20)]
    assert all(idx == 2 for idx in draws)
