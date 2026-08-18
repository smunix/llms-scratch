# Causal Attention and Transformer Block: Code Walkthrough

This guide explains the executable Chapter 4 package as a sequence of Candle tensor transformations. The package is intentionally compact but follows the GPT-style pre-layer-normalization block studied in the supplied chapter. [1] [2]

## Configuration

`TransformerConfig` centralizes the dimensions that must agree across every sublayer. The essential invariant is `embedding_dim % num_heads == 0`; it defines the per-head width.

```rust
let config = TransformerConfig {
    embedding_dim: 4,
    context_length: 4,
    num_heads: 2,
    feed_forward_multiplier: 2,
};
assert_eq!(config.head_dim()?, 2);
assert_eq!(config.feed_forward_dim()?, 8);
```

| Setting | Demo value | Role |
|---|---:|---|
| `embedding_dim` | 4 | Width entering and leaving the transformer block. |
| `context_length` | 4 | Largest sequence the causal mask permits. |
| `num_heads` | 2 | Number of independent attention subspaces. |
| `head_dim` | 2 | `embedding_dim / num_heads`. |
| `feed_forward_multiplier` | 2 | Compact teaching expansion factor; GPT-style blocks commonly use 4. |

## Layer normalization

`LayerNorm::identity(width)` creates affine parameters `scale = 1` and `shift = 0` as Candle tensors. The forward pass holds the final axis as a singleton dimension while computing the mean and variance, allowing the result to broadcast back over each token vector.

```rust
let mean = input.mean_keepdim(D::Minus1)?;
let centered = input.broadcast_sub(&mean)?;
let variance = centered.sqr()?.mean_keepdim(D::Minus1)?;
let normalized = centered.broadcast_div(&variance.affine(1.0, epsilon)?.sqrt()?)?;
let output = normalized.broadcast_mul(&scale)?.broadcast_add(&shift)?;
```

For an input shape `(B, T, d)`, every intermediate statistic has shape `(B, T, 1)`, while the normalized output remains `(B, T, d)`. This is why normalization does not mix information across examples or token positions. [1]

## Causal multi-head attention

`CausalMultiHeadAttention` first applies Q/K/V linear transformations through `broadcast_matmul` plus a broadcasted bias. Its `split_heads` helper reshapes `(B, T, d)` to `(B, T, H, d_head)` and transposes to `(B, H, T, d_head)`.

```rust
let scores = queries.matmul(&keys.transpose(2, 3)?)?
    .affine(1.0 / (head_dim as f64).sqrt(), 0.0)?;
let masked_scores = scores.broadcast_add(&causal_additive_mask(token_count, input.device())?)?;
let weights = softmax_last_dim(&masked_scores)?;
let per_head_context = weights.matmul(&values)?;
```

The additive mask contains negative infinity above the diagonal. Softmax assigns those entries probability zero, preventing future tokens from influencing the current token’s context vector. After attention, the module transposes and reshapes the heads back to `(B, T, d)` and applies the output projection. [1] [3]

```text
Head 0 causal pattern for T = 4
[•, 0, 0, 0]
[•, •, 0, 0]
[•, •, •, 0]
[•, •, •, •]
```

## Feed-forward network and GELU

`FeedForward::forward` applies an expand–activate–contract sequence independently at every token position. The first linear map takes `d` features to `multiplier × d`; Candle’s built-in `gelu` then supplies the GPT-2-style tanh approximation; the second linear map returns to `d`. [1] [2]

```rust
let expanded = linear(input, &expand_weight, &expand_bias)?;
let activated = gelu(&expanded)?;
let output = linear(&activated, &contract_weight, &contract_bias)?;
```

Because all operations broadcast across the batch and token axes, the feed-forward network does not permit cross-token visibility. It enriches the representation of each token only after the attention layer has selected contextual information.

## Pre-layer-normalized transformer block

The full transformer block has two residual steps. The implementation uses the intermediate `after_attention` tensor as the second shortcut source, matching the Chapter 4 ordering.

```rust
let attention_input = norm1.forward(input)?;
let attention_output = attention.forward(&attention_input)?;
let after_attention = input.broadcast_add(&attention_output)?;

let feed_forward_input = norm2.forward(&after_attention)?;
let feed_forward_output = feed_forward.forward(&feed_forward_input)?;
let output = after_attention.broadcast_add(&feed_forward_output)?;
```

| Stage | Input | Output | Cross-token mixing? |
|---|---|---|---|
| First LayerNorm | `(B, T, d)` | `(B, T, d)` | No. |
| Causal attention | `(B, T, d)` | `(B, T, d)` | Yes, but only current-and-past positions. |
| First residual | `(B, T, d)` | `(B, T, d)` | No additional mixing. |
| Second LayerNorm | `(B, T, d)` | `(B, T, d)` | No. |
| Feed-forward + GELU | `(B, T, d)` | `(B, T, d)` | No. |
| Second residual | `(B, T, d)` | `(B, T, d)` | No additional mixing. |

## Interpreting the runnable example

Run the executable with `cargo run --bin transformer_block`. It prints five shapes: the input, layer-normalized input, causal-attention weights, causal-attention output, and final transformer output. The final shape equals the input shape `(1, 4, 4)`, while the attention matrix has shape `(1, 2, 4, 4)` because it records a `4 × 4` query–key distribution for each of two heads.

The test suite adds a stronger causal check. It changes tokens at positions 2 and 3 while holding positions 0 and 1 fixed. The block outputs at positions 0 and 1 remain unchanged, which verifies that masking survives attention, residual, normalization, and feed-forward composition.

## References

[1] Sebastian Raschka, “Implementing a GPT Model from Scratch to Generate Text,” Chapter 4 in *Build a Large Language Model (From Scratch)*, Manning, 2025. [Official book page][1]

[2] Candle Core 0.6 API documentation. [Tensor API][2]

[3] Ashish Vaswani et al., “Attention Is All You Need,” 2017. [Paper][3]

[1]: https://www.manning.com/books/build-a-large-language-model-from-scratch
[2]: https://docs.rs/candle-core/0.6.0/candle_core/
[3]: https://arxiv.org/abs/1706.03762
