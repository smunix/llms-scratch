use candle_core::{Device, Tensor};
use chapter_03_attention::{
    causal_additive_mask, MultiHeadAttentionConfig, MultiHeadCausalAttention,
};
use nalgebra::DMatrix;
use pretty_assertions::assert_eq;

fn config() -> MultiHeadAttentionConfig {
    MultiHeadAttentionConfig {
        input_dim: 4,
        output_dim: 4,
        context_length: 4,
        num_heads: 2,
    }
}

fn input(values: &[f32]) -> Tensor {
    Tensor::from_vec(values.to_vec(), (1, 4, 4), &Device::Cpu).expect("input tensor is valid")
}

fn identity_attention() -> MultiHeadCausalAttention {
    let identity = DMatrix::identity(4, 4);
    MultiHeadCausalAttention::from_weight_matrices(
        config(),
        identity.clone(),
        identity.clone(),
        identity.clone(),
        identity,
    )
    .expect("identity attention configuration is valid")
}

#[test]
fn configuration_requires_output_dimension_to_split_evenly_across_heads() {
    let invalid = MultiHeadAttentionConfig {
        output_dim: 5,
        ..config()
    };
    assert_eq!(
        invalid
            .head_dim()
            .expect_err("5 cannot split across 2 heads"),
        "output_dim 5 must be divisible by num_heads 2"
    );
}

#[test]
fn causal_additive_mask_blocks_only_future_positions() {
    let mask = causal_additive_mask(4, &Device::Cpu)
        .expect("mask construction succeeds")
        .to_vec2::<f32>()
        .expect("mask materializes");
    assert_eq!(mask[0][0], 0.0);
    assert!(mask[0][1].is_infinite() && mask[0][1].is_sign_negative());
    assert_eq!(mask[3][0], 0.0);
    assert_eq!(mask[3][3], 0.0);
}

#[test]
fn multi_head_attention_preserves_expected_tensor_shapes_and_causal_probabilities() {
    let attention = MultiHeadCausalAttention::seeded(config(), 123).expect("model initializes");
    let trace = attention
        .forward_with_trace(&input(&[
            0.2, -0.1, 0.3, 0.0, 0.0, 0.5, -0.2, 0.1, 0.4, 0.1, 0.0, -0.3, -0.1, 0.2, 0.6, 0.2,
        ]))
        .expect("attention forward pass succeeds");

    assert_eq!(trace.queries.dims(), &[1, 2, 4, 2]);
    assert_eq!(trace.keys.dims(), &[1, 2, 4, 2]);
    assert_eq!(trace.values.dims(), &[1, 2, 4, 2]);
    assert_eq!(trace.attention_weights.dims(), &[1, 2, 4, 4]);
    assert_eq!(trace.per_head_context.dims(), &[1, 2, 4, 2]);
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
                    assert!(weight.abs() < 1e-7, "future attention weight must be zero");
                }
            });
            assert!(
                (row.iter().sum::<f32>() - 1.0).abs() < 1e-6,
                "each unmasked attention row must sum to one"
            );
        });
    });
}

#[test]
fn causal_outputs_for_a_prefix_do_not_change_when_future_tokens_change() {
    let attention = identity_attention();
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

    let before = attention
        .forward(&baseline)
        .expect("baseline forward pass")
        .to_vec3::<f32>()
        .expect("baseline output materializes");
    let after = attention
        .forward(&changed_future)
        .expect("modified forward pass")
        .to_vec3::<f32>()
        .expect("modified output materializes");

    (0..2).for_each(|position| {
        before[0][position]
            .iter()
            .zip(&after[0][position])
            .for_each(|(left, right)| {
                assert!(
                    (left - right).abs() < 1e-6,
                    "prefix outputs must not depend on future tokens"
                );
            });
    });
}
