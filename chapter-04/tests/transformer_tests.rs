use candle_core::{Device, Tensor};
use chapter_04_transformer::{
    gelu, CausalMultiHeadAttention, LayerNorm, TransformerBlock, TransformerConfig,
};
use pretty_assertions::assert_eq;

fn config() -> TransformerConfig {
    TransformerConfig {
        embedding_dim: 4,
        context_length: 4,
        num_heads: 2,
        feed_forward_multiplier: 2,
    }
}

fn input(values: &[f32]) -> Tensor {
    Tensor::from_vec(values.to_vec(), (1, 4, 4), &Device::Cpu).expect("input tensor is valid")
}

#[test]
fn configuration_requires_embedding_width_to_split_evenly_across_heads() {
    let invalid = TransformerConfig {
        embedding_dim: 5,
        ..config()
    };
    assert_eq!(
        invalid
            .head_dim()
            .expect_err("5 cannot split across 2 heads"),
        "embedding_dim 5 must be divisible by num_heads 2"
    );
}

#[test]
fn layer_norm_zero_centers_and_unit_normalizes_each_token_vector() {
    let norm = LayerNorm::identity(4).expect("layer norm initializes");
    let normalized = norm
        .forward(
            &Tensor::from_vec(
                vec![1.0_f32, 2.0, 3.0, 4.0, 4.0, 3.0, 2.0, 1.0],
                (1, 2, 4),
                &Device::Cpu,
            )
            .expect("input tensor is valid"),
        )
        .expect("normalization succeeds")
        .to_vec3::<f32>()
        .expect("normalized tensor materializes");

    normalized[0].iter().for_each(|row| {
        let mean = row.iter().sum::<f32>() / row.len() as f32;
        let variance =
            row.iter().map(|value| (value - mean).powi(2)).sum::<f32>() / row.len() as f32;
        assert!(mean.abs() < 1e-6, "normalized mean must be near zero");
        assert!(
            (variance - 1.0).abs() < 2e-4,
            "normalized variance must be near one"
        );
    });
}

#[test]
fn gelu_preserves_a_smooth_nonzero_negative_signal() {
    let output = gelu(
        &Tensor::from_vec(vec![-1.0_f32, 0.0, 1.0], 3, &Device::Cpu)
            .expect("activation input is valid"),
    )
    .expect("GELU succeeds")
    .to_vec1::<f32>()
    .expect("activation output materializes");
    assert!(output[0] < 0.0 && output[0] > -1.0);
    assert!(output[1].abs() < 1e-7);
    assert!(output[2] > 0.5 && output[2] < 1.0);
}

#[test]
fn causal_attention_has_expected_shapes_zero_future_weights_and_normalized_rows() {
    let attention = CausalMultiHeadAttention::seeded(config(), 123).expect("attention initializes");
    let trace = attention
        .forward_with_trace(&input(&[
            0.2, -0.1, 0.3, 0.0, 0.0, 0.5, -0.2, 0.1, 0.4, 0.1, 0.0, -0.3, -0.1, 0.2, 0.6, 0.2,
        ]))
        .expect("attention forward succeeds");
    assert_eq!(trace.attention_weights.dims(), &[1, 2, 4, 4]);
    assert_eq!(trace.output.dims(), &[1, 4, 4]);

    let weights = trace
        .attention_weights
        .flatten_all()
        .expect("weights flatten")
        .to_vec1::<f32>()
        .expect("weights materialize");
    (0..2).for_each(|head| {
        (0..4).for_each(|query_position| {
            let row_start = (head * 16) + (query_position * 4);
            let row = &weights[row_start..row_start + 4];
            row.iter().enumerate().for_each(|(key_position, weight)| {
                if key_position > query_position {
                    assert!(weight.abs() < 1e-7, "future attention must be zero");
                }
            });
            assert!((row.iter().sum::<f32>() - 1.0).abs() < 1e-6);
        });
    });
}

#[test]
fn transformer_block_preserves_shape_and_prefix_isolation() {
    let block = TransformerBlock::seeded(config(), 123).expect("transformer block initializes");
    let baseline = input(&[
        1.0, 0.0, 0.0, 0.0, // token 0
        0.0, 1.0, 0.0, 0.0, // token 1
        0.0, 0.0, 1.0, 0.0, // token 2
        0.0, 0.0, 0.0, 1.0, // token 3
    ]);
    let changed_future = input(&[
        1.0, 0.0, 0.0, 0.0, // unchanged token 0
        0.0, 1.0, 0.0, 0.0, // unchanged token 1
        50.0, -40.0, 30.0, -20.0, // altered future token 2
        -10.0, 20.0, -30.0, 40.0, // altered future token 3
    ]);

    let before = block
        .forward(&baseline)
        .expect("baseline transformer pass")
        .to_vec3::<f32>()
        .expect("baseline output materializes");
    let after = block
        .forward(&changed_future)
        .expect("modified transformer pass")
        .to_vec3::<f32>()
        .expect("modified output materializes");
    assert_eq!(before.len(), 1);
    assert_eq!(before[0].len(), 4);
    assert_eq!(before[0][0].len(), 4);

    (0..2).for_each(|position| {
        before[0][position]
            .iter()
            .zip(&after[0][position])
            .for_each(|(left, right)| {
                assert!(
                    (left - right).abs() < 2e-5,
                    "transformer prefix outputs must not depend on future tokens"
                );
            });
    });
}
