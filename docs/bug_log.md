# Bug Log

A running post-mortem of bugs encountered, root causes identified, and solutions implemented.

| Date | Phase | What Broke | Root Cause | Fix / Prevention |
| :--- | :--- | :--- | :--- | :--- |
| 2026-11-21 | Phase 0-1 | `pytest` failed collection with `ModuleNotFoundError: No module named 'mini_infer'` | Package was not installed in editable mode and `pyproject.toml` lacked pytest pythonpath configuration. | Added `pyproject.toml` with `[tool.pytest.ini_options] pythonpath = ["src"]` and installed via `pip install -e . --no-deps`. |
| 2026-11-21 | Phase 0-1 | `test_rope` failed with `TypeError: apply_rotary_pos_emb() got an unexpected keyword argument 'position_ids'` | In Transformers v5.x / 4.45+, `apply_rotary_pos_emb` API signature was streamlined to `(q, k, cos, sin)` without `position_ids`. | Updated test harness to call `hf_apply_rotary_pos_emb(q, k, cos, sin)`. |
| 2026-11-21 | Phase 0-1 | `test_grouped_query_attention` failed with `TypeError: cannot unpack non-iterable NoneType object` and return tuple mismatch | In Transformers v5.x, `LlamaAttention.forward` requires `position_embeddings=(cos, sin)` and returns a 2-tuple `(attn_output, attn_weights)` rather than 3-tuple. | Passed `position_embeddings=(cos_hf, sin_hf)` and unpacked 2-tuple `(hf_out, _)`. |
| 2026-11-21 | Phase 0-1 | `test_grouped_query_attention` failed numerical assertion (83.3% element mismatch, max diff 1.218) | Standalone `HFAttention` does not create a causal mask when `attention_mask=None` (HF assumes `LlamaModel` generates it). Our `GroupedQueryAttention` correctly defaults to applying a causal mask. Thus, HF was computing unmasked bidirectional attention while our module was computing causal attention. | Passed an explicit 4D upper-triangular causal mask to `hf_attn`. Outputs matched with 0.0 max absolute difference at `atol=1e-5`. |
