"""Unit tests verifying custom primitive layers against Hugging Face reference implementations."""

import torch
import pytest
from transformers.models.llama.modeling_llama import (
    LlamaRMSNorm as HFRMSNorm,
    LlamaMLP as HFMLP,
    LlamaAttention as HFAttention,
    apply_rotary_pos_emb as hf_apply_rotary_pos_emb,
)
from transformers import LlamaConfig as HFConfig

from mini_infer.config import LlamaConfig
from mini_infer.norm import RMSNorm
from mini_infer.rope import RotaryEmbedding, apply_rotary_pos_emb
from mini_infer.mlp import SwiGLUMLP
from mini_infer.attention import GroupedQueryAttention


@pytest.fixture
def smol_config():
    return LlamaConfig(
        vocab_size=49152,
        hidden_size=576,
        intermediate_size=1536,
        num_hidden_layers=2,  # Short 2 layers for quick unit tests
        num_attention_heads=9,
        num_key_value_heads=3,
        head_dim=64,
        rms_norm_eps=1e-5,
        rope_theta=10000.0,
        max_position_embeddings=2048,
    )


@pytest.fixture
def hf_config(smol_config):
    return HFConfig(
        vocab_size=smol_config.vocab_size,
        hidden_size=smol_config.hidden_size,
        intermediate_size=smol_config.intermediate_size,
        num_hidden_layers=smol_config.num_hidden_layers,
        num_attention_heads=smol_config.num_attention_heads,
        num_key_value_heads=smol_config.num_key_value_heads,
        head_dim=smol_config.head_dim,
        rms_norm_eps=smol_config.rms_norm_eps,
        rope_theta=smol_config.rope_theta,
        max_position_embeddings=smol_config.max_position_embeddings,
    )


def test_rms_norm(smol_config, hf_config):
    dim = smol_config.hidden_size
    custom_norm = RMSNorm(dim=dim, eps=smol_config.rms_norm_eps)
    hf_norm = HFRMSNorm(hidden_size=dim, eps=hf_config.rms_norm_eps)

    # Initialize weights identically
    weight = torch.randn(dim)
    custom_norm.weight.data.copy_(weight)
    hf_norm.weight.data.copy_(weight)

    x = torch.randn(2, 8, dim)
    custom_out = custom_norm(x)
    hf_out = hf_norm(x)

    torch.testing.assert_close(custom_out, hf_out, atol=1e-5, rtol=1e-5)


def test_rope(smol_config):
    head_dim = smol_config.head_dim
    seq_len = 16
    rotary = RotaryEmbedding(dim=head_dim, max_position_embeddings=smol_config.max_position_embeddings)

    q = torch.randn(2, smol_config.num_attention_heads, seq_len, head_dim)
    k = torch.randn(2, smol_config.num_key_value_heads, seq_len, head_dim)

    cos, sin = rotary(q, seq_len)
    custom_q, custom_k = apply_rotary_pos_emb(q, k, cos, sin)

    # Hugging Face reference apply_rotary_pos_emb (takes q, k, cos, sin)
    cos_hf = cos.squeeze(1)  # shape (1, seq_len, head_dim)
    sin_hf = sin.squeeze(1)
    hf_q, hf_k = hf_apply_rotary_pos_emb(q, k, cos_hf, sin_hf)

    torch.testing.assert_close(custom_q, hf_q, atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(custom_k, hf_k, atol=1e-5, rtol=1e-5)


def test_swiglu_mlp(smol_config, hf_config):
    custom_mlp = SwiGLUMLP(smol_config)
    hf_mlp = HFMLP(hf_config)

    # Copy weights
    with torch.no_grad():
        hf_mlp.gate_proj.weight.copy_(custom_mlp.gate_proj.weight)
        hf_mlp.up_proj.weight.copy_(custom_mlp.up_proj.weight)
        hf_mlp.down_proj.weight.copy_(custom_mlp.down_proj.weight)

    x = torch.randn(2, 8, smol_config.hidden_size)
    custom_out = custom_mlp(x)
    hf_out = hf_mlp(x)

    torch.testing.assert_close(custom_out, hf_out, atol=1e-5, rtol=1e-5)


def test_grouped_query_attention(smol_config, hf_config):
    custom_attn = GroupedQueryAttention(smol_config)
    hf_attn = HFAttention(config=hf_config, layer_idx=0)

    # Copy projection weights
    with torch.no_grad():
        hf_attn.q_proj.weight.copy_(custom_attn.q_proj.weight)
        hf_attn.k_proj.weight.copy_(custom_attn.k_proj.weight)
        hf_attn.v_proj.weight.copy_(custom_attn.v_proj.weight)
        hf_attn.o_proj.weight.copy_(custom_attn.o_proj.weight)

    seq_len = 6
    x = torch.randn(2, seq_len, smol_config.hidden_size)
    custom_out = custom_attn(x)
    
    # HF generates causal mask at model level; when testing attention directly,
    # pass 4D additive causal mask to HF layer
    causal_mask = torch.triu(
        torch.full((seq_len, seq_len), float("-inf")),
        diagonal=1,
    )[None, None, :, :]

    cos, sin = custom_attn.rotary_emb(x, seq_len)
    cos_hf = cos.squeeze(1)
    sin_hf = sin.squeeze(1)
    hf_out, _ = hf_attn(x, position_embeddings=(cos_hf, sin_hf), attention_mask=causal_mask)

    torch.testing.assert_close(custom_out, hf_out, atol=1e-5, rtol=1e-5)
