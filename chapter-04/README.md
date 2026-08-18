# Chapter 4 — Implementing a GPT Model from Scratch

**Source:** Sebastian Raschka, *Build a Large Language Model (From Scratch)*, Chapter 4, pp. 92–136. This is an original Rust and Candle implementation guide; it does **not** reproduce the supplied book or source code. [1]

> **Chapter thesis.** A GPT transformer block preserves the input embedding shape while combining causal multi-head attention, a position-wise feed-forward network, pre-layer normalization, and two residual connections. Repeating this block creates the contextual processing core of a GPT model. [1] [3]

## What this package implements

The Chapter 4 package is a standalone, **Candle-only** Rust project. It builds the transformer components that sit after the token and position embeddings of Chapter 2 and generalizes the causal attention mechanism from Chapter 3 into a pre-layer-normalization transformer block. Every matrix-like parameter, intermediate activation, mask, and output is a Candle tensor. [2]

| Chapter component | Rust API | Input → output shape |
|---|---|---|
| Configuration contract | `TransformerConfig` | Validates `d_model % num_heads == 0`. |
| Layer normalization | `LayerNorm` | `(B, T, d) → (B, T, d)`. |
| GELU | `gelu` | Preserves the input tensor shape. |
| Feed-forward network | `FeedForward` | `(B, T, d) → (B, T, 4d) → (B, T, d)`. |
| Causal multi-head attention | `CausalMultiHeadAttention` | `(B, T, d) → (B, T, d)`. |
| Transformer layer | `TransformerBlock` | `(B, T, d) → (B, T, d)`. |

## 1. Position in the GPT architecture

Chapter 2 turns text into position-aware token embeddings. Chapter 3 makes those embeddings context-aware through masked multi-head self-attention. Chapter 4 combines this attention sublayer with normalization, a nonlinear feed-forward transformation, and residual connections so that the same structure can be stacked repeatedly. A full GPT model adds token and position embedding tables before a sequence of these blocks, then applies final layer normalization and a vocabulary projection afterward. [1]

The package deliberately focuses on the reusable block rather than on the complete language-model head. This makes the architectural contract precise: the input and output must have the same shape, `B × T × d`, where `B` is batch size, `T` is sequence length, and `d` is model embedding width.

## 2. Causal multi-head attention

For an input `X`, the attention sublayer first computes learned query, key, and value projections. It splits the projected width into `H` heads of `d_head = d / H` values, performs scaled dot-product attention inside every head, combines the heads, and applies a final output projection. [1] [3]

```text
Q = XW_Q + b_Q                K = XW_K + b_K                V = XW_V + b_V
scores = (QKᵀ) / √d_head
weights = softmax(scores + causal_mask)
context = weights V
attention_output = concat(head_contexts) W_O + b_O
```

| Tensor | Shape after head split | Purpose |
|---|---|---|
| `Q`, `K`, `V` | `(B, H, T, d_head)` | Separate learned subspaces for querying, matching, and retrieving information. |
| Attention scores | `(B, H, T, T)` | One score for each query–key token pair within each head. |
| Attention weights | `(B, H, T, T)` | Row-normalized probabilities after causal masking. |
| Per-head context | `(B, H, T, d_head)` | Weighted sum of value vectors. |
| Attention output | `(B, T, d)` | Recombined and projected context vectors. |

`causal_additive_mask` creates a `T × T` tensor with zero at current-or-earlier positions and negative infinity at future positions. Adding it before softmax makes forbidden probabilities zero. The test suite verifies both properties: all weights above the diagonal are zero, and each valid row sums to one.

## 3. Layer normalization

Layer normalization operates independently on each token vector’s final embedding axis. It subtracts the mean, divides by the square root of variance plus a small epsilon, then applies a learned scale and shift. The implementation initializes the affine scale to one and shift to zero, producing the identity-affine version of normalization. [1]

```text
mean = mean(x, final axis)
variance = mean((x - mean)², final axis)
norm(x) = scale × (x - mean) / √(variance + ε) + shift
```

`LayerNorm::forward` expects a rank-three `(B, T, d)` Candle tensor. The normalization statistic retains the final axis with `mean_keepdim`, so Candle can broadcast the mean and denominator back across embedding values. The layer-normalization test checks that each normalized token vector has mean close to zero and variance close to one.

## 4. Feed-forward network with GELU

Attention mixes information across token positions. The feed-forward network instead transforms each token vector independently, using the same learned parameters at every sequence position. In the Chapter 4 GPT design, it expands the embedding width to four times its original size, applies GELU, and contracts back to `d`. [1]

```text
FFN(x) = GELU(xW_expand + b_expand)W_contract + b_contract
```

The package exposes the multiplier through `TransformerConfig::feed_forward_multiplier` so the examples can use a compact two-fold expansion while the chapter’s GPT-style default is conceptually four-fold. Candle’s built-in `Tensor::gelu()` supplies the tanh-based approximation used in GPT-2-style implementations. The GELU test confirms that a negative input retains a smooth, nonzero negative signal rather than being clamped to zero as with ReLU. [2]

## 5. Residual connections and pre-layer normalization

A residual, or shortcut, connection adds a sublayer’s input back to its output. It preserves the `B × T × d` shape and provides a direct route through a stack of blocks. In the pre-layer-normalization layout used in the chapter, normalization occurs before each major sublayer. [1]

```text
h = x + CausalMultiHeadAttention(LayerNorm₁(x))
y = h + FeedForward(LayerNorm₂(h))
```

This is exactly the sequence implemented by `TransformerBlock::forward`. The first residual joins the original input with causal attention output. The second joins that intermediate tensor with the feed-forward output. There is no hidden dropout in this educational inference package, so execution is deterministic; a training-oriented version would apply dropout after attention and feed-forward sublayers as the chapter describes.

| Transformer step | Candle operation | Shape |
|---|---|---|
| Normalize input | `LayerNorm::forward` | `(B, T, d)` |
| Causal attention | `CausalMultiHeadAttention::forward` | `(B, T, d)` |
| Attention residual | `broadcast_add` | `(B, T, d)` |
| Normalize intermediate | `LayerNorm::forward` | `(B, T, d)` |
| Expand, GELU, contract | `FeedForward::forward` | `(B, T, d)` |
| Feed-forward residual | `broadcast_add` | `(B, T, d)` |

## 6. Candle implementation contract

The demonstration config is intentionally small and readable:

```rust
let config = TransformerConfig {
    embedding_dim: 4,
    context_length: 4,
    num_heads: 2,
    feed_forward_multiplier: 2,
};
let block = TransformerBlock::seeded(config, 123)?;
let output = block.forward(&input)?;
assert_eq!(output.dims(), &[1, 4, 4]);
```

`seeded` uses deterministic Candle CPU tensors to make demonstrations reproducible. These tensors stand in for trained parameters. All forward operations are written against Candle tensor APIs—`broadcast_matmul`, `index`-free reshaping and transposition, `broadcast_add`, `mean_keepdim`, and built-in GELU—so the package has no dependency on a separate matrix library or Python runtime. [2]

## 7. Tests as architectural checks

| Test | Guarantee |
|---|---|
| `configuration_requires_embedding_width_to_split_evenly_across_heads` | The model width is divisible by its head count. |
| `layer_norm_zero_centers_and_unit_normalizes_each_token_vector` | Normalization is performed across each final embedding axis. |
| `gelu_preserves_a_smooth_nonzero_negative_signal` | GELU retains smooth negative activation behavior. |
| `causal_attention_has_expected_shapes_zero_future_weights_and_normalized_rows` | Masked attention has the expected tensor ranks, zero future attention, and normalized rows. |
| `transformer_block_preserves_shape_and_prefix_isolation` | The block preserves `B × T × d`, and changes to future tokens cannot affect earlier outputs. |

The prefix-isolation test is especially important. It changes later input token vectors substantially and checks that the outputs for the first two positions are unchanged. Layer normalization, feed-forward layers, and residual paths operate per position; the causal mask prevents the attention sublayer from becoming the path through which future positions could leak information backward.

## Running the package

From `chapter-04`, run:

```bash
cargo test --all-targets
cargo run --bin transformer_block
```

The executable reports the shapes at each major stage and prints one head’s attention matrix. The zero upper triangle provides a direct visual check of causal visibility, while the final output confirms that the transformer block preserves the input shape.

## Boundaries and next step

This package is an educational, deterministic forward-pass implementation. It deliberately omits parameter registration for optimization, gradient updates, training-mode dropout, GPU setup, batching utilities, vocabulary logits, embedding lookup, and multi-block GPT assembly. Those additions belong to a complete model and training pipeline. The implemented block nevertheless preserves the exact structural contract needed to stack transformer layers in a GPT architecture. [1] [2]

## References

[1] Sebastian Raschka, “Implementing a GPT Model from Scratch to Generate Text,” Chapter 4 in *Build a Large Language Model (From Scratch)*, pp. 92–136, Manning, 2025. [Official book page][1]

[2] Candle Core 0.6 API documentation. [Tensor API][2]

[3] Ashish Vaswani et al., “Attention Is All You Need,” 2017. [Paper][3]

[1]: https://www.manning.com/books/build-a-large-language-model-from-scratch
[2]: https://docs.rs/candle-core/0.6.0/candle_core/
[3]: https://arxiv.org/abs/1706.03762
