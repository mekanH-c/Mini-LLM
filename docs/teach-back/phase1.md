# Phase 1 Teach-Back: Naive LLaMA Architecture & Inference From Scratch

This document provides a comprehensive technical breakdown of Phase 0-1, written for software engineering and ML systems interviews.

---

## 1. What We Built and Why

In Phase 0-1, we constructed the core foundation of our inference engine: an end-to-end LLaMA decoder architecture built completely from raw PyTorch mathematical primitives, without wrapping Hugging Face's `LlamaForCausalLM` forward pass.

### Key Architectural Primitives:
1. **RMSNorm ([`norm.py`](file:///c:/Users/HP/Downloads/mini/src/mini_infer/norm.py))**:
   - *Why*: Standard LayerNorm computes both mean and variance $\mu, \sigma^2$. RMSNorm hypothesizes that the scaling property of LayerNorm is what stabilizes training, making mean-centering redundant. Dropping mean subtraction saves memory bandwidth and arithmetic operations with zero degradation in convergence.
   - *Formula*:
     $$\text{RMS}(x) = \sqrt{\frac{1}{d} \sum_{i=1}^d x_i^2 + \epsilon}, \quad y = \frac{x}{\text{RMS}(x)} \odot \gamma$$

2. **Rotary Position Embeddings (RoPE) ([`rope.py`](file:///c:/Users/HP/Downloads/mini/src/mini_infer/rope.py))**:
   - *Why*: Absolute positional embeddings (like in original Transformer or GPT-2) add vectors to token embeddings at the input layer. As tokens pass through dozens of non-linear layers, absolute position signal attenuates. RoPE instead injects position directly into the Query and Key vectors inside every attention layer via 2D rotation matrices, ensuring the dot product depends solely on relative token distances $(m - n)$.
   - *Implementation*: Slices $x_1 = x[..., :d/2], x_2 = x[..., d/2:]$; $\text{rotate\_half}(x) = [-x_2, x_1]$; $R(x) = (x \odot \cos) + (\text{rotate\_half}(x) \odot \sin)$.

3. **Grouped-Query Attention (GQA) ([`attention.py`](file:///c:/Users/HP/Downloads/mini/src/mini_infer/attention.py))**:
   - *Why*: In Multi-Head Attention (MHA), each query head has a dedicated key and value head ($H_Q = H_{KV} = 9$). In Multi-Query Attention (MQA), all query heads share 1 KV head ($H_{KV} = 1$). GQA strikes an optimal balance ($H_Q = 9, H_{KV} = 3$ in SmolLM-135M): $9 / 3 = 3$ query heads share each KV head. This slashes KV cache memory by $3\times$ while matching MHA generation quality.

4. **SwiGLU Multi-Layer Perceptron ([`mlp.py`](file:///c:/Users/HP/Downloads/mini/src/mini_infer/mlp.py))**:
   - *Why*: Traditional MLPs use $\text{GELU}(x W_1) W_2$. SwiGLU uses a GLU (Gated Linear Unit) variant:
     $$\text{SwiGLU}(x) = \left(\text{SiLU}(x W_{\text{gate}}) \odot (x W_{\text{up}})\right) W_{\text{down}}$$
     The element-wise gating allows the network to dynamically filter features, offering higher representational capacity per parameter.

5. **Direct `safetensors` Loader ([`loader.py`](file:///c:/Users/HP/Downloads/mini/src/mini_infer/loader.py))**:
   - Direct tensor ingestion from `model.safetensors` with an explicit key-mapping table (273 weights in SmolLM-135M) and weight tying verification (`lm_head.weight` = `model.embed_tokens.weight`).

6. **Sampling Strategies ([`sampling.py`](file:///c:/Users/HP/Downloads/mini/src/mini_infer/sampling.py))**:
   - Deterministic greedy (`argmax`), temperature scaling, top-$k$ filtering, and nucleus top-$p$ filtering via cumulative probability masking.

---

## 2. Complete Tensor Progression & Shapes

Using SmolLM-135M concrete dimensions:
- Batch size: $B$, Sequence length: $S$
- Hidden dimension $d_{\text{model}} = 576$
- Intermediate MLP dimension $d_{\text{ffn}} = 1536$
- Query heads $H_Q = 9$, Key/Value heads $H_{KV} = 3$, Head dimension $d_{\text{head}} = 64$
- Vocab size $V = 49{,}152$

```
Token IDs: (B, S)
    │
    ▼ Embedding Lookup
(B, S, 576)
    │
    ▼ ─── Repeated 30 Layers ───┐
    │                           │
    ├─► Pre-RMSNorm             │
    │   x: (B, S, 576)          │
    │                           │
    ├─► GroupedQueryAttention   │
    │   Q: (B, 9, S, 64)        │
    │   K: (B, 3, S, 64)        │
    │   V: (B, 3, S, 64)        │
    │   RoPE(Q, K)              │
    │   repeat_kv: K, V -> (B, 9, S, 64)
    │   Scores: Q @ K.T -> (B, 9, S, S)
    │   + Causal Mask & Softmax │
    │   Context: Weights @ V -> (B, 9, S, 64)
    │   Out Proj: (B, S, 576)   │
    │                           │
    ├─► Residual Add            │
    │                           │
    ├─► Pre-RMSNorm             │
    │                           │
    ├─► SwiGLU MLP              │
    │   Gate: (B, S, 1536)      │
    │   Up:   (B, S, 1536)      │
    │   Down: (B, S, 576)       │
    │                           │
    └─► Residual Add ───────────┘
    │
    ▼ Final RMSNorm: (B, S, 576)
    │
    ▼ LM Head Projection
Logits: (B, S, 49152)
```

---

## 3. Case Study: The `bfloat16` vs `float32` Divergence Bug

### What Broke:
When testing full-model logits against Hugging Face, all 4 primitive unit tests passed with $10^{-5}$ precision, but full-model logits drifted by **1.401** max absolute difference.

### The Systematic Debugging Process:
Following Rule 3 (*"stop adding features and compare layer by layer"*), we wrote [`compare_layer_by_layer.py`](file:///c:/Users/HP/Downloads/mini/scripts/compare_layer_by_layer.py):
1. **Embedding Check**: `(custom_emb - hf_emb).abs().max() == 0.0` (Weights loaded identically).
2. **Layer Inspection**: Discovered PyTorch threw a type error in isolated layer comparisons:
   ```
   Expected query, key, and value to have the same dtype, but got query.dtype: float and value.dtype: BFloat16
   ```
3. **Root Cause**: SmolLM-135M was trained and stored on HF in `bfloat16`. `transformers.AutoModelForCausalLM.from_pretrained` automatically loaded the reference model in `torch.bfloat16`. Our custom loader defaulted to `torch.float32`.
   - `bfloat16` has only 7 bits of mantissa (fractional precision ~ $10^{-2}$ to $10^{-3}$).
   - Evaluating 30 layers in `float32` vs `bfloat16` causes small rounding errors to compound exponentially through consecutive non-linear layers, resulting in ~1.4 logit divergence.
4. **The Fix & Proof**:
   When both models were loaded with matching precision (`torch.float32`):
   - Per-layer isolated difference across all 30 layers: $\le 6.1 \times 10^{-5}$
   - Final RMSNorm difference: $2.38 \times 10^{-5}$
   - Full forward logits difference: **$5.15 \times 10^{-5}$** ($0.000051$, strictly below the $10^{-4}$ threshold).

---

## 4. 3 High-Yield Interview Questions & Answers

### Q1: Why does modern LLM architecture (LLaMA, SmolLM, Mistral) prefer GQA over MHA or MQA? What is the computational and memory trade-off?
**Answer**:
During training (prefill), attention is compute-bound, and Multi-Head Attention (MHA) achieves high parallel efficiency. However, during autoregressive decoding, generating one token at a time requires loading the entire historical Key and Value tensors from high-bandwidth memory (HBM) into SRAM for every single token step. This makes decoding **memory-bandwidth bound**.
- In **MHA** ($H_Q = H_{KV}$), the KV cache size is $2 \times L \times H_Q \times d_{\text{head}} \times S \times \text{bytes}$. For high concurrency, GPU VRAM fills rapidly with KV caches rather than compute batches.
- In **MQA** ($H_{KV} = 1$), cache size is minimized by an order of magnitude, but model quality suffers on complex reasoning and retrieval tasks because all query heads must share identical key representations.
- **GQA** partitions query heads into $G$ groups ($H_Q / H_{KV}$). In SmolLM-135M, $9$ query heads share $3$ KV heads (ratio $3:1$). This slashes KV cache memory footprint and memory-bandwidth traffic by **$3\times$** while maintaining nearly 100% of MHA accuracy.

### Q2: How does RoPE encode relative positional distance without adding absolute positional embedding vectors? Explain the mathematical mechanism.
**Answer**:
RoPE operates by applying a 2D rotation to pairs of coordinate dimensions within the Query and Key vectors:
$$\mathbf{R}_{\Theta, m}^d = \text{diag}\left( R_{\theta_1, m}, R_{\theta_2, m}, \dots, R_{\theta_{d/2}, m} \right)$$
where each $R_{\theta_i, m}$ is a 2D rotation matrix by angle $m \theta_i$:
$$R_{\theta_i, m} = \begin{pmatrix} \cos(m \theta_i) & -\sin(m \theta_i) \\ \sin(m \theta_i) & \cos(m \theta_i) \end{pmatrix}$$

The key mathematical property is that rotation matrices are orthogonal ($R^T R = I$) and commutative:
$$\langle R_m \mathbf{q}, R_n \mathbf{k} \rangle = (R_m \mathbf{q})^T (R_n \mathbf{k}) = \mathbf{q}^T R_m^T R_n \mathbf{k} = \mathbf{q}^T R_{n - m} \mathbf{k}$$
The inner product (which determines attention score before softmax) depends purely on the relative distance $(n - m)$, not on the absolute positions $m$ or $n$. Furthermore, because rotation preserves vector magnitude ($||R_m \mathbf{q}|| = ||\mathbf{q}||$), RoPE avoids distorting the norm of query and key activations.

### Q3: Why is naive autoregressive decoding computationally inefficient, and how do we quantify the FLOPs wasted on every token step without a KV cache?
**Answer**:
In naive autoregressive decoding, to generate token $t+1$, the engine passes the entire sequence $x_{1:t}$ through all layers of the model.
- At step $t$, the linear projections for Query, Key, and Value are recomputed for all $t$ tokens from scratch:
  $$\text{FLOPs}_{\text{proj}}(t) = 2 \times t \times d_{\text{model}} \times (d_Q + d_K + d_V + d_O)$$
- Across generating $N$ tokens starting from prompt length $P$, total projection FLOPs scale as:
  $$\sum_{t=P}^{P+N-1} O(t) = O(N \cdot P + N^2)$$
  The earlier tokens $x_1, \dots, x_P$ are projected through $W_q, W_k, W_v$ $N$ separate times, completely redundantly.
- Furthermore, calculating attention scores requires re-evaluating $Q_{1:t} K_{1:t}^T$, leading to quadratic complexity $O(N^3)$ across generation.
- A **KV Cache** resolves this: because previous tokens' Key and Value states are invariant in causal attention, we compute $K_t$ and $V_t$ once for the newest token only ($O(1)$ projection), concatenate them to the cache, and perform matrix-vector attention against the cache.
