# Metrics Log

This log tracks measured performance metrics across each phase of development. 

> **Important**: No metrics are fabricated. Every entry corresponds to an actual benchmark run, recording hardware details, library versions, model, and configuration parameters.

| Date | Phase / Change | Hardware / Device | Model | Prompt Len / Gen Len | Batch Size | Tokens / sec | TTFT (ms) | Peak Memory | Notes |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| 2026-11-21 | Phase 0-1 Custom Naive Baseline | Intel x64 CPU (12 cores) | SmolLM-135M (float32) | avg ~6 / 32 | 1 | 8.42 | 92.26 | 1427.4 MB RSS | Full-sequence recomputation per token (no KV cache). PyTorch 2.15.0+cpu. |
| 2026-11-21 | Phase 0-1 Hugging Face Reference | Intel x64 CPU (12 cores) | SmolLM-135M (float32) | avg ~6 / 32 | 1 | 15.64 | 77.51 | 1423.2 MB RSS | HF `generate()` on same 5 prompts (uses internal KV cache). Transformers 5.23.0. |
| 2026-11-22 | Phase 0-1 Custom Naive Baseline | NVIDIA Tesla T4 (Kaggle) | SmolLM-135M (float32) | avg ~6 / 32 | 1 | 27.58 | 35.08 | 573.67 MB VRAM | Custom naive engine (isolated GPU memory run). PyTorch 2.11.0+cu130, Transformers 5.20.2, Safetensors 0.8.0. 15 runs averaged (5 prompts x 3 runs). An earlier run yielded 29.88 tokens/s (~8% run-to-run variation); both engines are statistically tied at short 32-token generation length because GPU kernel launch overhead dominates compute. |
| 2026-11-22 | Phase 0-1 Hugging Face Reference | NVIDIA Tesla T4 (Kaggle) | SmolLM-135M (float32) | avg ~6 / 32 | 1 | 26.53 | 42.36 | 533.37 MB VRAM | HF `generate()` reference (isolated GPU memory run). PyTorch 2.11.0+cu130, Transformers 5.20.2, Safetensors 0.8.0. 15 runs averaged (5 prompts x 3 runs). Statistically tied with custom naive engine at 32 new tokens due to kernel overhead dominance on small models. |
