import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from mini_infer.loader import load_from_hf

MODEL_ID = "HuggingFaceTB/SmolLM-135M"

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
hf_model = AutoModelForCausalLM.from_pretrained(MODEL_ID, torch_dtype=torch.float32).eval()
custom_model, _ = load_from_hf(MODEL_ID, dtype=torch.float32)
custom_model.eval()

prompt = "The quick brown fox jumps over the lazy dog."
input_ids = tokenizer(prompt, return_tensors="pt")["input_ids"]

with torch.no_grad():
    # 1. Embedding comparison
    custom_emb = custom_model.model.embed_tokens(input_ids)
    hf_emb = hf_model.model.embed_tokens(input_ids)
    print("Embedding diff:", (custom_emb - hf_emb).abs().max().item())

    # 2. Check position embeddings
    seq_len = input_ids.shape[1]
    cos_c, sin_c = custom_model.model.layers[0].self_attn.rotary_emb(custom_emb, seq_len)
    
    # 3. Layer-by-layer comparison
    # In HF 5.x, prepare causal mask and position embeddings
    # Let's inspect HF model forward
    from transformers.masking_utils import create_causal_mask
    try:
        causal_mask = create_causal_mask(input_ids, custom_emb, 0)
    except Exception:
        # Fallback to standard 4D upper triangular
        causal_mask = torch.triu(
            torch.full((seq_len, seq_len), float("-inf")), diagonal=1
        )[None, None, :, :]

    cos_hf = cos_c.squeeze(1)
    sin_hf = sin_c.squeeze(1)

    custom_hidden = custom_emb
    hf_hidden = hf_emb

    for idx, (custom_layer, hf_layer) in enumerate(zip(custom_model.model.layers, hf_model.model.layers)):
        # Feed SAME input into both layers to isolate layer implementation
        custom_out_isolated = custom_layer(hf_hidden)
        hf_out = hf_layer(
            hf_hidden,
            position_embeddings=(cos_hf, sin_hf),
            attention_mask=causal_mask,
        )
        if isinstance(hf_out, tuple):
            hf_out_isolated = hf_out[0]
        else:
            hf_out_isolated = hf_out

        isolated_diff = (custom_out_isolated - hf_out_isolated).abs().max().item()

        # Feed previous layer's output (cumulative error)
        custom_hidden = custom_layer(custom_hidden)
        hf_out_cum = hf_layer(
            hf_hidden,
            position_embeddings=(cos_hf, sin_hf),
            attention_mask=causal_mask,
        )
        hf_hidden = hf_out_cum[0] if isinstance(hf_out_cum, tuple) else hf_out_cum
        cumulative_diff = (custom_hidden - hf_hidden).abs().max().item()

        print(f"Layer {idx:2d}: isolated_diff={isolated_diff:.6e} | cumulative_diff={cumulative_diff:.6e} | shape={hf_hidden.shape}")

    # Final norm
    c_final = custom_model.model.norm(custom_hidden)
    h_final = hf_model.model.norm(hf_hidden)
    print("Final norm diff:", (c_final - h_final).abs().max().item())

    # LM Head
    c_logits = custom_model.lm_head(c_final)
    h_logits = hf_model.lm_head(h_final)
    print("LM Head logits diff:", (c_logits - h_logits).abs().max().item())

    # Full forward pass call directly
    c_full_logits = custom_model(input_ids)
    h_full_logits = hf_model(input_ids).logits
    print("Direct full-model forward logits diff:", (c_full_logits - h_full_logits).abs().max().item())
