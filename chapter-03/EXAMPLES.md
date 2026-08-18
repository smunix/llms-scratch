# Multi-Head Causal Attention: Code Walkthrough

This guide maps the executable Rust code to the Chapter 3 attention calculation. The implementation uses two complementary libraries: `nalgebra` provides explicit, inspectable dense matrices for deterministic Q/K/V and output weights, while Candle performs the rank-3 and rank-4 tensor operations of the actual attention pass. [1] [2]

## Configuration and shape invariant

The module begins with a configuration. `output_dim` is the combined output width, so it must divide evenly into the requested number of heads. The per-head width is `head_dim = output_dim / num_heads`.

```rust
let config = MultiHeadAttentionConfig {
    input_dim: 4,
    output_dim: 4,
    context_length: 4,
    num_heads: 2,
};
assert_eq!(config.head_dim()?, 2);
```

| Quantity | Value in the executable | Interpretation |
|---|---:|---|
| `B` | 1 | One input sequence in the demonstration batch. |
| `T` | 4 | Four input tokens. |
| `d_in` | 4 | Four values in each input embedding. |
| `d_out` | 4 | Four values in each final output embedding. |
| `H` | 2 | Two independent attention heads. |
| `d_head` | 2 | Two dimensions processed by each head. |

## Deterministic weights and Candle conversion

`MultiHeadCausalAttention::seeded` makes four deterministic `DMatrix<f32>` values: query, key, value, and output weights. It then converts each matrix to a Candle CPU tensor in row-major order. The seeded route exists solely to make tests and examples repeatable. Training later replaces these with learnable parameter tensors.

The constructor `from_weight_matrices` is particularly useful for experiments. It validates that every Q/K/V matrix has shape `d_in × d_out` and that the output matrix has shape `d_out × d_out`. The prefix-isolation test uses identity matrices through this constructor, removing random projections from the causality proof.

## Projection and head split

The input to `forward_with_trace` must have shape `(B, T, d_in)`. `broadcast_matmul` applies each shared Q/K/V matrix across the batch and token axes, yielding `(B, T, d_out)`. Each result then passes through `split_heads`:

```rust
projection
    .reshape((batch_size, token_count, num_heads, head_dim))?
    .transpose(1, 2)?
```

The first line makes the head axis explicit; the second brings it before the token axis. The result has `(B, H, T, d_head)`, so ordinary batched matrix multiplication now computes every head’s attention scores without an explicit Rust loop.

## Scaled dot-product attention and causal mask

For every batch item and head, queries multiply transposed keys to produce a `T × T` score matrix. The code divides by `sqrt(d_head)`, then applies the additive mask before softmax:

```rust
let scores = queries.matmul(&keys.transpose(2, 3)?)?
    .affine(1.0 / (head_dim as f64).sqrt(), 0.0)?;
let masked_scores = scores.broadcast_add(&causal_additive_mask(token_count, input.device())?)?;
let attention_weights = softmax_last_dim(&masked_scores)?;
```

`causal_additive_mask` returns `0` at and below the diagonal and `-∞` above it. Candle broadcasts the `T × T` mask across `B` and `H`. The stable softmax implementation subtracts the last-axis maximum, exponentiates, sums with `keepdim`, and divides. Therefore, the valid weights on each row sum to one and forbidden future weights are zero. [1] [2]

## Combine heads and project

The weights multiply values to create `(B, H, T, d_head)` per-head context vectors. Then the module transposes back, makes the result contiguous, flattens the head and per-head-width axes to `d_out`, and applies the output matrix:

```rust
let combined = per_head_context
    .transpose(1, 2)?
    .contiguous()?
    .reshape((batch_size, token_count, output_dim))?;
let output = combined.broadcast_matmul(&output_weight)?;
```

The final output shape is `(B, T, d_out)`. The output projection is a standard part of the efficient multi-head attention implementation: it combines information after the head outputs have been concatenated. [1]

## Reading the executable output

Run `cargo run --bin multi_head_attention`. Its attention-weight matrix for head 0 has the pattern below, where `•` represents a learned nonzero probability and `0` is an enforced causal zero.

```text
[•, 0, 0, 0]
[•, •, 0, 0]
[•, •, •, 0]
[•, •, •, •]
```

The reported row sums are all approximately 1.0. This is a direct observable check that the mask was applied before the softmax and that the attention weights remain normalized.

## Extension map

| Desired extension | Insertion point | Main concern |
|---|---|---|
| Q/K/V bias | Add bias tensors after each projection. | Bias must broadcast over `B` and `T`. |
| Attention dropout | Apply a stochastic mask after softmax. | Use only during training and rescale retained weights. |
| Batch size greater than one | Pass a larger first dimension. | The current tensor operations already broadcast correctly. |
| Longer context | Increase `context_length` and pass `T ≤ context_length`. | Attention memory grows quadratically with `T`. |
| KV cache for generation | Cache previous K and V tensors per layer. | Preserve the causal position offset. |

## References

[1] Sebastian Raschka, “Coding Attention Mechanisms,” Chapter 3 in *Build a Large Language Model (From Scratch)*, Manning, 2025. [Official book page][1]

[2] Candle Core 0.6 API documentation. [Tensor API][2]

[1]: https://www.manning.com/books/build-a-large-language-model-from-scratch
[2]: https://docs.rs/candle-core/0.6.0/candle_core/
