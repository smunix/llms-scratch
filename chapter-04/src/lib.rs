//! Chapter 4 transformer components implemented with Candle tensors only.
//!
//! The central block uses the GPT-style pre-layer-normalization sequence:
//! `x -> x + causal_attention(norm1(x)) -> x + feed_forward(norm2(x))`.
//! All examples are deterministic CPU inference demonstrations; trainable parameter
//! registration and dropout are deliberate next steps rather than hidden behavior.

use candle_core::{Device, Tensor, D};
use itertools::Itertools;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TransformerConfig {
    pub embedding_dim: usize,
    pub context_length: usize,
    pub num_heads: usize,
    pub feed_forward_multiplier: usize,
}

impl TransformerConfig {
    pub fn head_dim(&self) -> Result<usize, String> {
        if self.embedding_dim == 0 || self.context_length == 0 || self.feed_forward_multiplier == 0
        {
            return Err("embedding_dim, context_length, and feed_forward_multiplier must be greater than zero".to_owned());
        }
        if self.num_heads == 0 {
            return Err("num_heads must be greater than zero".to_owned());
        }
        (self.embedding_dim % self.num_heads == 0)
            .then_some(self.embedding_dim / self.num_heads)
            .ok_or_else(|| {
                format!(
                    "embedding_dim {} must be divisible by num_heads {}",
                    self.embedding_dim, self.num_heads
                )
            })
    }

    pub fn feed_forward_dim(&self) -> Result<usize, String> {
        self.head_dim()?;
        self.embedding_dim
            .checked_mul(self.feed_forward_multiplier)
            .ok_or_else(|| "feed-forward width overflowed usize".to_owned())
    }
}

/// Learnable affine layer-normalization tensors initialized to identity scale and zero shift.
#[derive(Debug, Clone)]
pub struct LayerNorm {
    width: usize,
    epsilon: f64,
    scale: Tensor,
    shift: Tensor,
}

impl LayerNorm {
    pub fn identity(width: usize) -> Result<Self, String> {
        if width == 0 {
            return Err("layer-normalization width must be greater than zero".to_owned());
        }
        let device = Device::Cpu;
        let scale = Tensor::from_vec(vec![1.0_f32; width], width, &device)
            .map_err(|error| format!("could not create layer-norm scale tensor: {error}"))?;
        let shift = Tensor::from_vec(vec![0.0_f32; width], width, &device)
            .map_err(|error| format!("could not create layer-norm shift tensor: {error}"))?;
        Self::from_affine(width, scale, shift, 1e-5)
    }

    pub fn from_affine(
        width: usize,
        scale: Tensor,
        shift: Tensor,
        epsilon: f64,
    ) -> Result<Self, String> {
        if width == 0 || epsilon <= 0.0 {
            return Err(
                "layer-normalization width and epsilon must be greater than zero".to_owned(),
            );
        }
        [(&scale, "scale"), (&shift, "shift")]
            .into_iter()
            .try_for_each(|(tensor, name)| {
                let observed = tensor
                    .dims1()
                    .map_err(|error| format!("layer-norm {name} must be rank 1: {error}"))?;
                (observed == width).then_some(()).ok_or_else(|| {
                    format!("layer-norm {name} width {observed} does not match {width}")
                })
            })?;
        Ok(Self {
            width,
            epsilon,
            scale,
            shift,
        })
    }

    /// Normalize each token vector across its final embedding axis.
    pub fn forward(&self, input: &Tensor) -> Result<Tensor, String> {
        let (_, _, width) = input
            .dims3()
            .map_err(|error| format!("layer norm expects input shape (B, T, d): {error}"))?;
        if width != self.width {
            return Err(format!(
                "layer-norm input width {width} does not match configured width {}",
                self.width
            ));
        }
        let mean = input
            .mean_keepdim(D::Minus1)
            .map_err(|error| format!("layer-norm mean failed: {error}"))?;
        let centered = input
            .broadcast_sub(&mean)
            .map_err(|error| format!("layer-norm centering failed: {error}"))?;
        let variance = centered
            .sqr()
            .map_err(|error| format!("layer-norm square failed: {error}"))?
            .mean_keepdim(D::Minus1)
            .map_err(|error| format!("layer-norm variance failed: {error}"))?;
        let denominator = variance
            .affine(1.0, self.epsilon)
            .map_err(|error| format!("layer-norm epsilon addition failed: {error}"))?
            .sqrt()
            .map_err(|error| format!("layer-norm square root failed: {error}"))?;
        centered
            .broadcast_div(&denominator)
            .map_err(|error| format!("layer-norm division failed: {error}"))?
            .broadcast_mul(&self.scale)
            .map_err(|error| format!("layer-norm scale failed: {error}"))?
            .broadcast_add(&self.shift)
            .map_err(|error| format!("layer-norm shift failed: {error}"))
    }
}

/// The GPT-2-style tanh approximation of the Gaussian error linear unit.
pub fn gelu(input: &Tensor) -> Result<Tensor, String> {
    input
        .gelu()
        .map_err(|error| format!("GELU activation failed: {error}"))
}

#[derive(Debug, Clone)]
pub struct FeedForward {
    input_width: usize,
    hidden_width: usize,
    expand_weight: Tensor,
    expand_bias: Tensor,
    contract_weight: Tensor,
    contract_bias: Tensor,
}

impl FeedForward {
    pub fn seeded(input_width: usize, multiplier: usize, seed: u64) -> Result<Self, String> {
        if input_width == 0 || multiplier == 0 {
            return Err(
                "feed-forward input width and multiplier must be greater than zero".to_owned(),
            );
        }
        let hidden_width = input_width
            .checked_mul(multiplier)
            .ok_or_else(|| "feed-forward hidden width overflowed usize".to_owned())?;
        let device = Device::Cpu;
        let (seed, expand_weight) = seeded_tensor(input_width, hidden_width, seed, &device)?;
        let (seed, expand_bias) = seeded_tensor_1d(hidden_width, seed, &device)?;
        let (seed, contract_weight) = seeded_tensor(hidden_width, input_width, seed, &device)?;
        let (_, contract_bias) = seeded_tensor_1d(input_width, seed, &device)?;
        Ok(Self {
            input_width,
            hidden_width,
            expand_weight,
            expand_bias,
            contract_weight,
            contract_bias,
        })
    }

    pub fn forward(&self, input: &Tensor) -> Result<Tensor, String> {
        let (_, _, width) = input
            .dims3()
            .map_err(|error| format!("feed-forward expects input shape (B, T, d): {error}"))?;
        if width != self.input_width {
            return Err(format!(
                "feed-forward input width {width} does not match configured width {}",
                self.input_width
            ));
        }
        let expanded = linear(input, &self.expand_weight, &self.expand_bias)?;
        let activated = gelu(&expanded)?;
        linear(&activated, &self.contract_weight, &self.contract_bias)
    }

    pub fn hidden_width(&self) -> usize {
        self.hidden_width
    }
}

/// Intermediate tensors exposed for inspecting causal attention in a learning setting.
#[derive(Debug, Clone)]
pub struct CausalAttentionTrace {
    /// Shape `(B, H, T, T)` after causal masking and softmax.
    pub attention_weights: Tensor,
    /// Shape `(B, T, d)` after output projection.
    pub output: Tensor,
}

/// Efficient, masked multi-head attention with one combined Q/K/V projection per type.
#[derive(Debug, Clone)]
pub struct CausalMultiHeadAttention {
    config: TransformerConfig,
    query_weight: Tensor,
    query_bias: Tensor,
    key_weight: Tensor,
    key_bias: Tensor,
    value_weight: Tensor,
    value_bias: Tensor,
    output_weight: Tensor,
    output_bias: Tensor,
}

impl CausalMultiHeadAttention {
    pub fn seeded(config: TransformerConfig, seed: u64) -> Result<Self, String> {
        config.head_dim()?;
        let device = Device::Cpu;
        let width = config.embedding_dim;
        let (seed, query_weight) = seeded_tensor(width, width, seed, &device)?;
        let (seed, query_bias) = seeded_tensor_1d(width, seed, &device)?;
        let (seed, key_weight) = seeded_tensor(width, width, seed, &device)?;
        let (seed, key_bias) = seeded_tensor_1d(width, seed, &device)?;
        let (seed, value_weight) = seeded_tensor(width, width, seed, &device)?;
        let (seed, value_bias) = seeded_tensor_1d(width, seed, &device)?;
        let (seed, output_weight) = seeded_tensor(width, width, seed, &device)?;
        let (_, output_bias) = seeded_tensor_1d(width, seed, &device)?;
        Ok(Self {
            config,
            query_weight,
            query_bias,
            key_weight,
            key_bias,
            value_weight,
            value_bias,
            output_weight,
            output_bias,
        })
    }

    pub fn forward(&self, input: &Tensor) -> Result<Tensor, String> {
        Ok(self.forward_with_trace(input)?.output)
    }

    pub fn forward_with_trace(&self, input: &Tensor) -> Result<CausalAttentionTrace, String> {
        let (batch_size, token_count, width) = input
            .dims3()
            .map_err(|error| format!("causal attention expects input shape (B, T, d): {error}"))?;
        if width != self.config.embedding_dim {
            return Err(format!(
                "attention input width {width} does not match configured embedding_dim {}",
                self.config.embedding_dim
            ));
        }
        if token_count > self.config.context_length {
            return Err(format!(
                "token count {token_count} exceeds context length {}",
                self.config.context_length
            ));
        }
        let head_dim = self.config.head_dim()?;
        let queries = split_heads(
            linear(input, &self.query_weight, &self.query_bias)?,
            batch_size,
            token_count,
            self.config.num_heads,
            head_dim,
        )?;
        let keys = split_heads(
            linear(input, &self.key_weight, &self.key_bias)?,
            batch_size,
            token_count,
            self.config.num_heads,
            head_dim,
        )?;
        let values = split_heads(
            linear(input, &self.value_weight, &self.value_bias)?,
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
            .reshape((batch_size, token_count, self.config.embedding_dim))
            .map_err(|error| format!("head combination reshape failed: {error}"))?;
        let output = linear(&combined, &self.output_weight, &self.output_bias)?;
        Ok(CausalAttentionTrace {
            attention_weights,
            output,
        })
    }
}

/// A GPT-style pre-layer-norm transformer block with two residual connections.
#[derive(Debug, Clone)]
pub struct TransformerBlock {
    config: TransformerConfig,
    norm1: LayerNorm,
    attention: CausalMultiHeadAttention,
    norm2: LayerNorm,
    feed_forward: FeedForward,
}

impl TransformerBlock {
    pub fn seeded(config: TransformerConfig, seed: u64) -> Result<Self, String> {
        config.feed_forward_dim()?;
        Ok(Self {
            norm1: LayerNorm::identity(config.embedding_dim)?,
            attention: CausalMultiHeadAttention::seeded(config, seed)?,
            norm2: LayerNorm::identity(config.embedding_dim)?,
            feed_forward: FeedForward::seeded(
                config.embedding_dim,
                config.feed_forward_multiplier,
                seed.wrapping_add(0x9e37_79b9),
            )?,
            config,
        })
    }

    /// Execute `x + attention(norm1(x))`, followed by `x + feed_forward(norm2(x))`.
    pub fn forward(&self, input: &Tensor) -> Result<Tensor, String> {
        let (_, token_count, width) = input
            .dims3()
            .map_err(|error| format!("transformer block expects input shape (B, T, d): {error}"))?;
        if width != self.config.embedding_dim || token_count > self.config.context_length {
            return Err(format!(
                "transformer input shape {:?} is incompatible with embedding_dim {} and context_length {}",
                input.dims(), self.config.embedding_dim, self.config.context_length
            ));
        }
        let attention_input = self.norm1.forward(input)?;
        let attention_output = self.attention.forward(&attention_input)?;
        let after_attention = input
            .broadcast_add(&attention_output)
            .map_err(|error| format!("attention residual addition failed: {error}"))?;
        let feed_forward_input = self.norm2.forward(&after_attention)?;
        let feed_forward_output = self.feed_forward.forward(&feed_forward_input)?;
        after_attention
            .broadcast_add(&feed_forward_output)
            .map_err(|error| format!("feed-forward residual addition failed: {error}"))
    }
}

/// Additive mask with zero at legal current-or-past positions and negative infinity in the future.
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

fn linear(input: &Tensor, weight: &Tensor, bias: &Tensor) -> Result<Tensor, String> {
    input
        .broadcast_matmul(weight)
        .map_err(|error| format!("linear projection matmul failed: {error}"))?
        .broadcast_add(bias)
        .map_err(|error| format!("linear projection bias addition failed: {error}"))
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
    let (state, values) = seeded_values(rows * columns, seed);
    let tensor = Tensor::from_vec(values, (rows, columns), device)
        .map_err(|error| format!("could not initialize seeded weight tensor: {error}"))?;
    Ok((state, tensor))
}

fn seeded_tensor_1d(length: usize, seed: u64, device: &Device) -> Result<(u64, Tensor), String> {
    let (state, values) = seeded_values(length, seed);
    let tensor = Tensor::from_vec(values, length, device)
        .map_err(|error| format!("could not initialize seeded bias tensor: {error}"))?;
    Ok((state, tensor))
}

fn seeded_values(length: usize, seed: u64) -> (u64, Vec<f32>) {
    (0..length).fold(
        (seed, Vec::with_capacity(length)),
        |(state, mut values), _| {
            let next = state.wrapping_mul(6364136223846793005).wrapping_add(1);
            let unit = ((next >> 32) as f32) / (u32::MAX as f32);
            values.push(unit - 0.5);
            (next, values)
        },
    )
}
