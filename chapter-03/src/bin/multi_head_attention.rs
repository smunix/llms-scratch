use candle_core::{Device, Tensor};
use chapter_03_attention::{MultiHeadAttentionConfig, MultiHeadCausalAttention};
use itertools::Itertools;

fn main() -> Result<(), String> {
    let config = MultiHeadAttentionConfig {
        input_dim: 4,
        output_dim: 4,
        context_length: 4,
        num_heads: 2,
    };
    let attention = MultiHeadCausalAttention::seeded(config, 123)?;
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
    let trace = attention.forward_with_trace(&input)?;
    let flattened_weights = trace
        .attention_weights
        .flatten_all()
        .map_err(|error| format!("could not flatten attention weights: {error}"))?
        .to_vec1::<f32>()
        .map_err(|error| format!("could not inspect attention weights: {error}"))?;

    println!("Input shape: {:?}", input.dims());
    println!("Q/K/V shape after head split: {:?}", trace.queries.dims());
    println!(
        "Attention-weight shape: {:?}",
        trace.attention_weights.dims()
    );
    println!(
        "Per-head context shape: {:?}",
        trace.per_head_context.dims()
    );
    println!("Final output shape: {:?}", trace.output.dims());

    let first_head = flattened_weights[..16]
        .chunks(4)
        .map(|row| row.to_vec())
        .collect_vec();
    println!("\nHead 0 causal attention weights:");
    first_head.iter().for_each(|row| println!("  {row:?}"));
    let row_sums = first_head
        .iter()
        .map(|row| row.iter().sum::<f32>())
        .collect_vec();
    println!("Head 0 row sums: {row_sums:?}");

    println!(
        "\nEvery position can attend to itself and earlier tokens only; entries above the causal diagonal are zero."
    );
    Ok(())
}
