# Chapter 3 — Coding Attention Mechanisms

**Source:** Sebastian Raschka, *Build a Large Language Model (From Scratch)*, Chapter 3, pp. 50–91. This is an original Rust and Candle study guide; it does **not** reproduce the supplied book or source code. [1]

> **Chapter thesis.** Attention turns a token representation into a context-aware representation by using a weighted combination of value vectors. In a GPT-style model, causal masking forbids information flow from future positions, and multi-head attention repeats the calculation in several learned subspaces at once. [1] [3]

## What this package implements

The Chapter 3 package implements the chapter’s efficient multi-head attention design entirely with Candle CPU tensors. Candle owns deterministic projection-weight initialization, batched projections, head splitting, scaled dot products, causal masking, softmax, head recombination, and output projection. [2]

| Chapter concept | Rust API | Key tensor shape |
|---|---|---|
| Multi-head configuration | `MultiHeadAttentionConfig` | Stores `d_in`, `d_out`, context length, and head count. |
| Q/K/V and output matrices | `MultiHeadCausalAttention` | Each Q/K/V matrix is `d_in × d_out`; output matrix is `d_out × d_out`. |
| Causal mask | `causal_additive_mask` | `T × T`, with `-∞` strictly above the diagonal. |
| Efficient head splitting | `split_heads` | `(B, T, d_out) → (B, H, T, d_head)`. |
| Causal multi-head forward pass | `forward` | `(B, T, d_in) → (B, T, d_out)`. |
| Inspectable forward pass | `forward_with_trace` | Retains Q, K, V, attention weights, per-head context, and output. |

## 1. The motivation for self-attention

A token’s useful meaning often depends on other tokens in its context. Self-attention lets every position form a context vector by assigning a weight to each input position and taking a weighted sum of values. The chapter builds from a basic dot-product intuition toward **scaled dot-product attention** with learned query, key, and value projections. [1]

For one attention head, let `X` be an input sequence, `W_Q`, `W_K`, and `W_V` be learned linear projections, and `d_head` be the per-head width. The central calculation is:

```text
Q = X W_Q
K = X W_K
V = X W_V
scores = (Q Kᵀ) / √d_head
weights = softmax(scores)
context = weights V
```

The scale by `√d_head` stabilizes the magnitude of dot-product scores as the head width grows. The softmax makes each query position’s weights a probability distribution across allowable key positions, so every row sums to one. [1] [3]

| Symbol | Meaning | Shape in this implementation |
|---|---|---|
| `B` | Batch size | Batch axis. |
| `T` | Number of tokens in the supplied sequence | Must be no greater than `context_length`. |
| `d_in` | Width of input token embeddings | Final axis of `X`. |
| `d_out` | Total output width across all heads | Final axis after head recombination. |
| `H` | Number of attention heads | Must divide `d_out`. |
| `d_head` | Per-head width, `d_out / H` | Final axis while heads are separated. |

## 2. Causal attention prevents future-token leakage

Standard self-attention lets every token look at every other token. That is invalid for left-to-right language modeling: when predicting the token after a prefix, a model must not use a token it has not yet generated. Chapter 3 therefore applies a **causal mask** that keeps the diagonal and lower triangle but disables the upper triangle. [1]

This module represents the mask additively. It creates a `T × T` tensor with zero at legal positions and `-∞` where `key_position > query_position`. Adding it to the scaled scores before softmax makes every forbidden probability exactly zero:

```text
masked_scores[q, k] = scores[q, k] + mask[q, k]
mask[q, k] = 0       if k ≤ q
mask[q, k] = -∞      if k > q
```

Applying the mask **before** softmax is important. Since `exp(-∞) = 0`, masked positions contribute nothing to either the numerator or denominator of softmax. Thus, every output position depends only on its current-and-earlier prefix. The integration test changes tokens 2 and 3 dramatically and verifies that outputs at positions 0 and 1 remain unchanged.

## 3. Why multi-head attention exists

A single attention head has one learned query/key/value subspace. Multi-head attention gives the model several independent subspaces, allowing different heads to emphasize different positional or representational relationships. The intuitive implementation runs several causal-attention modules and concatenates their outputs. Chapter 3 then uses a more efficient equivalent: one large Q/K/V projection followed by reshaping and batched matrix multiplication. [1]

The efficient approach used here first projects to `d_out`, then splits that final axis:

```text
Q, K, V: (B, T, d_out)
reshape: (B, T, H, d_head)
transpose: (B, H, T, d_head)
```

Each head now owns a `T × d_head` query/key/value slice. Candle performs the four-dimensional matrix multiplication across every batch item and head in parallel. After contextualization, the module reverses the transformation:

```text
per-head context: (B, H, T, d_head)
transpose:        (B, T, H, d_head)
reshape:          (B, T, d_out)
output projection: (B, T, d_out)
```

The `output_weight` matrix mixes the concatenated head results. It is customary in GPT-style transformer blocks and ensures that information from multiple heads can interact after concatenation. [1]

## 4. Candle implementation walkthrough

The essential configuration checks appear before any tensor algebra. The implementation rejects zero dimensions and requires that `d_out` is exactly divisible by `num_heads`.

```rust
let config = MultiHeadAttentionConfig {
    input_dim: 4,
    output_dim: 4,
    context_length: 4,
    num_heads: 2,
};
assert_eq!(config.head_dim()?, 2);
```

`MultiHeadCausalAttention::seeded` creates deterministic Candle CPU tensors for experimentation. `from_weight_tensors` accepts explicit Candle Q/K/V and output tensors, validates their rank and shapes, and supports controlled experiments without leaving the tensor runtime. Production training would replace this deterministic initialization with trainable parameter tensors. The forward pass uses `broadcast_matmul` for linear projections; this shares a `d_in × d_out` weight tensor across the batch axis. [2]

```rust
let trace = attention.forward_with_trace(&input)?;
assert_eq!(trace.queries.dims(), &[1, 2, 4, 2]);
assert_eq!(trace.attention_weights.dims(), &[1, 2, 4, 4]);
assert_eq!(trace.output.dims(), &[1, 4, 4]);
```

The trace is not necessary for a production forward pass, but it makes the chapter’s intermediate shapes inspectable. Use `forward` when only the final `(B, T, d_out)` context tensor is required.

## 5. Tests as mathematical guardrails

| Test | Guarantee |
|---|---|
| `configuration_requires_output_dimension_to_split_evenly_across_heads` | Invalid head layouts are rejected before computation. |
| `causal_additive_mask_blocks_only_future_positions` | The upper triangle is negative infinity; the diagonal and lower triangle remain zero. |
| `multi_head_attention_preserves_expected_tensor_shapes_and_causal_probabilities` | Q/K/V, weights, contexts, and output have the expected rank and size; future weights are zero and every row sums to one. |
| `causal_outputs_for_a_prefix_do_not_change_when_future_tokens_change` | Earlier output positions are isolated from altered future input tokens. |

These tests verify more than shapes. In particular, the prefix-isolation test captures the essential functional promise of causal attention: mutating an unseen suffix cannot alter the contextual representation of an earlier position.

## Running the code

From this directory, run:

```bash
cargo test --all-targets
cargo run --bin multi_head_attention
```

The executable reports the input, Q/K/V, score-weight, per-head-context, and output tensor shapes. It prints one head’s `4 × 4` attention-weight matrix and its row sums so the causal zero pattern and normalization can be inspected directly.

## Boundaries and next step

The module implements deterministic **inference-style** causal attention. It intentionally omits trainable parameter registration, bias terms, dropout, GPU selection, key/value caching, mixed precision, and automatic differentiation wiring. Candle provides the tensor primitives required for those later extensions, but their correct training-loop integration belongs to subsequent chapters. [2]

The next architectural step is a transformer block: layer normalization, feed-forward layers, residual connections, and this multi-head attention module combine into the GPT model described in Chapter 4.

## References

[1] Sebastian Raschka, “Coding Attention Mechanisms,” Chapter 3 in *Build a Large Language Model (From Scratch)*, pp. 50–91, Manning, 2025. [Official book page][1]

[2] Candle Core 0.6 API documentation. [Tensor API][2]

[3] Ashish Vaswani et al., “Attention Is All You Need,” 2017. [Paper][3]

[1]: https://www.manning.com/books/build-a-large-language-model-from-scratch
[2]: https://docs.rs/candle-core/0.6.0/candle_core/
[3]: https://arxiv.org/abs/1706.03762
