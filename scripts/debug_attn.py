import torch
from transformers import LlamaConfig as HFConfig
from transformers.models.llama.modeling_llama import LlamaAttention as HFAttention
from mini_infer.config import LlamaConfig
from mini_infer.attention import GroupedQueryAttention

smol_config = LlamaConfig(
    vocab_size=49152,
    hidden_size=576,
    intermediate_size=1536,
    num_hidden_layers=2,
    num_attention_heads=9,
    num_key_value_heads=3,
    head_dim=64,
    rms_norm_eps=1e-5,
    rope_theta=10000.0,
    max_position_embeddings=2048,
)

hf_config = HFConfig(
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
    _attn_implementation="eager",
)

custom_attn = GroupedQueryAttention(smol_config)
hf_attn = HFAttention(config=hf_config, layer_idx=0)

# Copy weights
with torch.no_grad():
    hf_attn.q_proj.weight.copy_(custom_attn.q_proj.weight)
    hf_attn.k_proj.weight.copy_(custom_attn.k_proj.weight)
    hf_attn.v_proj.weight.copy_(custom_attn.v_proj.weight)
    hf_attn.o_proj.weight.copy_(custom_attn.o_proj.weight)

seq_len = 6
x = torch.randn(2, seq_len, smol_config.hidden_size)

# Custom steps
q_custom = custom_attn.q_proj(x).view(2, seq_len, 9, 64).transpose(1, 2)
k_custom = custom_attn.k_proj(x).view(2, seq_len, 3, 64).transpose(1, 2)
v_custom = custom_attn.v_proj(x).view(2, seq_len, 3, 64).transpose(1, 2)

cos, sin = custom_attn.rotary_emb(x, seq_len)
cos_hf = cos.squeeze(1)
sin_hf = sin.squeeze(1)

# HF steps
hidden_shape = (2, seq_len, -1, 64)
q_hf = hf_attn.q_proj(x).view(hidden_shape).transpose(1, 2)
k_hf = hf_attn.k_proj(x).view(hidden_shape).transpose(1, 2)
v_hf = hf_attn.v_proj(x).view(hidden_shape).transpose(1, 2)

print("Q diff:", (q_custom - q_hf).abs().max().item())
print("K diff:", (k_custom - k_hf).abs().max().item())
print("V diff:", (v_custom - v_hf).abs().max().item())

from mini_infer.rope import apply_rotary_pos_emb
from transformers.models.llama.modeling_llama import apply_rotary_pos_emb as hf_apply_rotary_pos_emb

q_rot, k_rot = apply_rotary_pos_emb(q_custom, k_custom, cos, sin)
q_hf_rot, k_hf_rot = hf_apply_rotary_pos_emb(q_hf, k_hf, cos_hf, sin_hf)

print("Q rot diff:", (q_rot - q_hf_rot).abs().max().item())
print("K rot diff:", (k_rot - k_hf_rot).abs().max().item())

# HF forward pass check
hf_out, _ = hf_attn(x, position_embeddings=(cos_hf, sin_hf))
custom_out = custom_attn(x)
print("Out diff:", (custom_out - hf_out).abs().max().item())

# Let's see what HF's attention_mask does inside HFAttention
from transformers.modeling_attn_mask_utils import _prepare_4d_causal_attention_mask
causal_mask = _prepare_4d_causal_attention_mask(None, (2, seq_len), x, 0)
print("causal_mask shape:", causal_mask.shape if causal_mask is not None else None)
hf_out_masked, _ = hf_attn(x, position_embeddings=(cos_hf, sin_hf), attention_mask=causal_mask)
print("Out diff with HF causal_mask:", (custom_out - hf_out_masked).abs().max().item())
