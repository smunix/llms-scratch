//! Chapter 3 attention components implemented in Rust.
//!
//! The implementation keeps the matrix shapes from the chapter explicit:
//! `input (B, T, d_in) -> Q/K/V (B, H, T, d_head) -> output (B, T, d_out)`.
//! It is an educational, inference-oriented module implemented entirely with Candle CPU tensors.

use candle_core::{Device, Tensor, D};
use itertools::Itertools;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MultiHeadAttentionConfig {
    pub input_dim: usize,
    pub output_dim: usize,
    pub context_length: usize,
    pub num_heads: usize,
}

impl MultiHeadAttentionConfig {
    pub fn head_dim(&self) -> Result<usize, String> {
        if self.input_dim == 0 || self.output_dim == 0 || self.context_length == 0 {
            return Err(
                "input_dim, output_dim, and context_length must all be greater than zero"
                    .to_owned(),
            );
        }
        if self.num_heads == 0 {
            return Err("num_heads must be greater than zero".to_owned());
        }
        (self.output_dim % self.num_heads == 0)
            .then_some(self.output_dim / self.num_heads)
            .ok_or_else(|| {
                format!(
                    "output_dim {} must be divisible by num_heads {}",
                    self.output_dim, self.num_heads
                )
            })
    }
}

/// Intermediate tensors exposed for inspection in a learning setting.
#[derive(Debug, Clone)]
pub struct AttentionTrace {
    /// Shape `(B, H, T, d_head)` after Q/K/V projection, reshape, and head transpose.
    pub queries: Tensor,
    pub keys: Tensor,
    pub values: Tensor,
    /// Shape `(B, H, T, T)` after scaling, causal masking, and softmax.
    pub attention_weights: Tensor,
    /// Shape `(B, H, T, d_head)` before heads are combined.
    pub per_head_context: Tensor,
    /// Shape `(B, T, d_out)` after combining heads and output projection.
    pub output: Tensor,
}

/// Efficient GPT-style multi-head **causal** attention.
///
/// The three Q/K/V projections are each computed once at `output_dim`, then reshaped into
/// `num_heads` independent subspaces. This is equivalent to separate per-head projections but
/// enables one batched matmul for all heads. The module has no bias and uses no dropout so its
/// trace is deterministic and easy to test; dropout can be inserted after `attention_weights`
/// during training.
#[derive(Debug, Clone)]
pub struct MultiHeadCausalAttention {
    config: MultiHeadAttentionConfig,
    query_weight: Tensor,
    key_weight: Tensor,
    value_weight: Tensor,
    output_weight: Tensor,
}

impl MultiHeadCausalAttention {
    /// Construct deterministic Candle CPU tensors for the demonstration weights.
    /// The values are trainable parameters in a full training system.
    pub fn seeded(config: MultiHeadAttentionConfig, seed: u64) -> Result<Self, String> {
        config.head_dim()?;
        let device = Device::Cpu;
        let (seed, query) = seeded_tensor(config.input_dim, config.output_dim, seed, &device)?;
        let (seed, key) = seeded_tensor(config.input_dim, config.output_dim, seed, &device)?;
        let (seed, value) = seeded_tensor(config.input_dim, config.output_dim, seed, &device)?;
        let (_, output) = seeded_tensor(config.output_dim, config.output_dim, seed, &device)?;
        Self::from_weight_tensors(config, query, key, value, output)
    }

    /// Construct attention from explicit Candle weight tensors. This makes controlled tests and
    /// experiments possible without leaving the tensor runtime.
    pub fn from_weight_tensors(
        config: MultiHeadAttentionConfig,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        output: Tensor,
    ) -> Result<Self, String> {
        config.head_dim()?;
        let expected_qkv = (config.input_dim, config.output_dim);
        let expected_output = (config.output_dim, config.output_dim);
        [(&query, "query"), (&key, "key"), (&value, "value")]
            .into_iter()
            .try_for_each(|(tensor, name)| {
                let shape = tensor
                    .dims2()
                    .map_err(|error| format!("{name} weight must be rank 2: {error}"))?;
                (shape == expected_qkv).then_some(()).ok_or_else(|| {
                    format!("{name} weight has shape {shape:?}; expected {expected_qkv:?}")
                })
            })?;
        let output_shape = output
            .dims2()
            .map_err(|error| format!("output weight must be rank 2: {error}"))?;
        (output_shape == expected_output)
            .then_some(())
            .ok_or_else(|| {
                format!("output weight has shape {output_shape:?}; expected {expected_output:?}")
            })?;

        Ok(Self {
            config,
            query_weight: query,
            key_weight: key,
            value_weight: value,
            output_weight: output,
        })
    }

    pub fn config(&self) -> MultiHeadAttentionConfig {
        self.config
    }

    /// Return only the final projected context vectors with shape `(B, T, d_out)`.
    pub fn forward(&self, input: &Tensor) -> Result<Tensor, String> {
        Ok(self.forward_with_trace(input)?.output)
    }

    /// Compute attention while retaining the key intermediate tensors for inspection and tests.
    pub fn forward_with_trace(&self, input: &Tensor) -> Result<AttentionTrace, String> {
        let (batch_size, token_count, input_dim) = input
            .dims3()
            .map_err(|error| format!("attention expects input with shape (B, T, d_in): {error}"))?;
        if input_dim != self.config.input_dim {
            return Err(format!(
                "input embedding width {input_dim} does not match configured input_dim {}",
                self.config.input_dim
            ));
        }
        if token_count > self.config.context_length {
            return Err(format!(
                "token count {token_count} exceeds configured context_length {}",
                self.config.context_length
            ));
        }

        let head_dim = self.config.head_dim()?;
        let queries = split_heads(
            input
                .broadcast_matmul(&self.query_weight)
                .map_err(|error| format!("query projection failed: {error}"))?,
            batch_size,
            token_count,
            self.config.num_heads,
            head_dim,
        )?;
        let keys = split_heads(
            input
                .broadcast_matmul(&self.key_weight)
                .map_err(|error| format!("key projection failed: {error}"))?,
            batch_size,
            token_count,
            self.config.num_heads,
            head_dim,
        )?;
        let values = split_heads(
            input
                .broadcast_matmul(&self.value_weight)
                .map_err(|error| format!("value projection failed: {error}"))?,
            batch_size,
            token_count,
            self.config.num_heads,
            head_dim,
        )?;

        let scores = queries
            .matmul(
                &keys
                    .transpose(2, 3)
                    .map_err(|error| format!("key transpose failed: {error}"))?,
            )
            .map_err(|error| format!("attention-score matmul failed: {error}"))?
            .affine(1.0 / (head_dim as f64).sqrt(), 0.0)
            .map_err(|error| format!("attention scaling failed: {error}"))?;
        let masked_scores = scores
            .broadcast_add(&causal_additive_mask(token_count, input.device())?)
            .map_err(|error| format!("causal-mask application failed: {error}"))?;
        let attention_weights = softmax_last_dim(&masked_scores)?;
        let per_head_context = attention_weights
            .matmul(&values)
            .map_err(|error| format!("weighted-value matmul failed: {error}"))?;
        let combined = per_head_context
            .transpose(1, 2)
            .map_err(|error| format!("head transpose failed: {error}"))?
            .contiguous()
            .map_err(|error| format!("head tensor could not be made contiguous: {error}"))?
            .reshape((batch_size, token_count, self.config.output_dim))
            .map_err(|error| format!("head combination reshape failed: {error}"))?;
        let output = combined
            .broadcast_matmul(&self.output_weight)
            .map_err(|error| format!("output projection failed: {error}"))?;

        Ok(AttentionTrace {
            queries,
            keys,
            values,
            attention_weights,
            per_head_context,
            output,
        })
    }
}

/// Make an additive causal mask with zero on and below the diagonal and negative infinity above it.
/// The mask broadcasts from `(T, T)` across the batch and head dimensions of attention scores.
pub fn causal_additive_mask(token_count: usize, device: &Device) -> Result<Tensor, String> {
    let values = (0..token_count)
        .flat_map(|query_position| {
            (0..token_count).map(move |key_position| {
                (key_position > query_position)
                    .then_some(f32::NEG_INFINITY)
                    .unwrap_or(0.0)
            })
        })
        .collect_vec();
    Tensor::from_vec(values, (token_count, token_count), device)
        .map_err(|error| format!("could not create causal mask: {error}"))
}

fn split_heads(
    projection: Tensor,
    batch_size: usize,
    token_count: usize,
    num_heads: usize,
    head_dim: usize,
) -> Result<Tensor, String> {
    projection
        .reshape((batch_size, token_count, num_heads, head_dim))
        .map_err(|error| format!("Q/K/V head split reshape failed: {error}"))?
        .transpose(1, 2)
        .map_err(|error| format!("Q/K/V head split transpose failed: {error}"))
}

fn softmax_last_dim(scores: &Tensor) -> Result<Tensor, String> {
    let maxima = scores
        .max_keepdim(D::Minus1)
        .map_err(|error| format!("softmax maximum failed: {error}"))?;
    let exponentials = scores
        .broadcast_sub(&maxima)
        .map_err(|error| format!("softmax centering failed: {error}"))?
        .exp()
        .map_err(|error| format!("softmax exponential failed: {error}"))?;
    let denominator = exponentials
        .sum_keepdim(D::Minus1)
        .map_err(|error| format!("softmax denominator failed: {error}"))?;
    exponentials
        .broadcast_div(&denominator)
        .map_err(|error| format!("softmax normalization failed: {error}"))
}

fn seeded_tensor(
    rows: usize,
    columns: usize,
    seed: u64,
    device: &Device,
) -> Result<(u64, Tensor), String> {
    let (state, values) = (0..(rows * columns)).fold(
        (seed, Vec::with_capacity(rows * columns)),
        |(state, mut values), _| {
            let next = state.wrapping_mul(6364136223846793005).wrapping_add(1);
            let unit = ((next >> 32) as f32) / (u32::MAX as f32);
            values.push(unit - 0.5);
            (next, values)
        },
    );
    let tensor = Tensor::from_vec(values, (rows, columns), device)
        .map_err(|error| format!("could not initialize Candle weight tensor: {error}"))?;
    Ok((state, tensor))
}
