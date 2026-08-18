use candle_core::{Device, Tensor};
use chapter_04_transformer::{
    CausalMultiHeadAttention, LayerNorm, TransformerBlock, TransformerConfig,
};
use itertools::Itertools;

fn main() -> Result<(), String> {
    let config = TransformerConfig {
        embedding_dim: 4,
        context_length: 4,
        num_heads: 2,
        feed_forward_multiplier: 2,
    };
    let input = Tensor::from_vec(
        vec![
            0.2_f32, -0.1, 0.3, 0.0, // token 0
            0.0, 0.5, -0.2, 0.1, // token 1
            0.4, 0.1, 0.0, -0.3, // token 2
            -0.1, 0.2, 0.6, 0.2, // token 3
        ],
        (1, 4, 4),
        &Device::Cpu,
    )
    .map_err(|error| format!("could not create demo input: {error}"))?;

    let normalized = LayerNorm::identity(config.embedding_dim)?.forward(&input)?;
    let attention = CausalMultiHeadAttention::seeded(config, 123)?;
    let attention_trace = attention.forward_with_trace(&normalized)?;
    let block = TransformerBlock::seeded(config, 123)?;
    let output = block.forward(&input)?;

    let flattened_weights = attention_trace
        .attention_weights
        .flatten_all()
        .map_err(|error| format!("could not flatten attention weights: {error}"))?
        .to_vec1::<f32>()
        .map_err(|error| format!("could not inspect attention weights: {error}"))?;
    let first_head = flattened_weights[..16]
        .chunks(4)
        .map(|row| row.to_vec())
        .collect_vec();

    println!("Input shape: {:?}", input.dims());
    println!("LayerNorm output shape: {:?}", normalized.dims());
    println!(
        "Causal attention weight shape: {:?}",
        attention_trace.attention_weights.dims()
    );
    println!(
        "Causal attention output shape: {:?}",
        attention_trace.output.dims()
    );
    println!("Transformer-block output shape: {:?}", output.dims());
    println!("\nHead 0 causal attention weights:");
    first_head.iter().for_each(|row| println!("  {row:?}"));
    println!(
        "\nThe transformer block preserves (batch, tokens, embedding width) while using pre-layer normalization and two residual additions."
    );
    Ok(())
}
