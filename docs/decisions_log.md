# Decisions Log

This document records architectural, technical, and algorithmic decisions made throughout the project, detailing the options considered, trade-offs evaluated, and reasons for each choice.

| Date | Decision | Options Considered | Rationale / Trade-offs | Phase |
| :--- | :--- | :--- | :--- | :--- |
| 2026-11-20 | Project structure & execution environment | Mono-repo package (`src/mini_infer`) with modular phase-based progression. CPU-first for dev with Kaggle/Colab T4 notebooks for GPU/Triton. | Allows rapid iterative development and unit testing on local laptop, keeping code accessible and interview-ready. | Setup |
| 2026-11-20 | Target reference model | SmolLM-135M (`HuggingFaceTB/SmolLM-135M`) as primary, TinyLlama-1.1B as secondary. | SmolLM-135M is a modern LLaMA architecture (GQA, RoPE, RMSNorm, SwiGLU) that runs swiftly on CPU (~135M parameters, float32/bfloat16 weights ~270MB) while maintaining state-of-the-art token quality and identical architectural primitives to modern production LLMs. | Phase 0-1 |
| 2026-11-20 | Direct `safetensors` weight loading | 1. Hugging Face `from_pretrained` (blackbox forward pass)<br>2. Direct `safetensors` file parsing into custom tensors | Option 2 fulfills the project objective of building from scratch. We map tensor names explicitly to our custom weights without relying on HF's model classes. | Phase 0-1 |
| 2026-11-21 | Standalone Attention Causal Masking | 1. Rely on model-level mask passing only<br>2. Default to internal causal mask when `attention_mask is None` | Option 2 makes `GroupedQueryAttention` safe and self-contained when used in isolation or unit tests, preventing accidental unmasked bidirectional attention. | Phase 0-1 |
| 2026-11-21 | Numerical precision alignment in tests | 1. Test in checkpoint native `bfloat16`<br>2. Test in explicit `torch.float32` | Option 2 prevents 16-bit mantissa roundoff from accumulating across 30 deep layers, ensuring tight $10^{-5}$ tolerance verification against reference models. | Phase 0-1 |
| 2026-11-21 | RoPE Cos/Sin Caching Strategy | 1. Compute trigonometric frequencies on-the-fly<br>2. Precompute tables up to `max_position_embeddings` as non-persistent buffers | Option 2 eliminates repeated transcendental function calls during generation while keeping memory overhead negligible (~1MB). | Phase 0-1 |
