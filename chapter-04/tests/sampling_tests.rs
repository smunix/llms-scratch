use candle_core::{Device, Tensor};
use chapter_04_transformer::{
    temperature_distribution, top_k_distribution, top_p_distribution, SamplingStrategy,
    TokenSampler,
};
use pretty_assertions::assert_eq;

fn logits(values: Vec<f32>) -> Tensor {
    let length = values.len();
    Tensor::from_vec(values, length, &Device::Cpu).expect("logit tensor is valid")
}

fn probability_values(distribution: &chapter_04_transformer::FilteredDistribution) -> Vec<f32> {
    distribution
        .probabilities
        .to_vec1::<f32>()
        .expect("probabilities materialize")
}

#[test]
fn lower_temperature_concentrates_probability_on_the_largest_logit() {
    let source = logits(vec![2.0, 1.0, 0.0]);
    let warm = temperature_distribution(&source, 2.0).expect("warm distribution");
    let cool = temperature_distribution(&source, 0.5).expect("cool distribution");
    let warm_values = probability_values(&warm);
    let cool_values = probability_values(&cool);

    assert!((warm_values.iter().sum::<f32>() - 1.0).abs() < 1e-6);
    assert!((cool_values.iter().sum::<f32>() - 1.0).abs() < 1e-6);
    assert!(cool_values[0] > warm_values[0]);
    assert!(cool_values[2] < warm_values[2]);
}

#[test]
fn top_k_keeps_exactly_the_highest_k_logits_and_renormalizes() {
    let distribution =
        top_k_distribution(&logits(vec![1.0, 4.0, 3.0, 2.0]), 1.0, 2).expect("top-k distribution");
    let probabilities = probability_values(&distribution);

    assert_eq!(distribution.retained_token_ids, vec![1, 2]);
    assert_eq!(probabilities[0], 0.0);
    assert!(probabilities[1] > 0.0);
    assert!(probabilities[2] > 0.0);
    assert_eq!(probabilities[3], 0.0);
    assert!((probabilities.iter().sum::<f32>() - 1.0).abs() < 1e-6);
}

#[test]
fn top_p_keeps_the_smallest_descending_probability_prefix_reaching_threshold() {
    let distribution =
        top_p_distribution(&logits(vec![2.0, 1.0, 0.0]), 1.0, 0.8).expect("top-p distribution");
    let probabilities = probability_values(&distribution);

    assert_eq!(distribution.retained_token_ids, vec![0, 1]);
    assert!(probabilities[0] > 0.0);
    assert!(probabilities[1] > 0.0);
    assert_eq!(probabilities[2], 0.0);
    assert!((probabilities.iter().sum::<f32>() - 1.0).abs() < 1e-6);
}

#[test]
fn greedy_and_seeded_top_k_sampling_select_only_permitted_token_ids() {
    let source = logits(vec![0.1, 2.0, 1.0, -1.0]);
    let mut greedy_sampler = TokenSampler::seeded(7);
    let greedy = greedy_sampler
        .sample(&source, SamplingStrategy::Greedy)
        .expect("greedy sampling");
    assert_eq!(greedy.token_id, 1);
    assert_eq!(
        probability_values(&greedy.distribution),
        vec![0.0, 1.0, 0.0, 0.0]
    );

    let mut first = TokenSampler::seeded(99);
    let mut second = TokenSampler::seeded(99);
    let first_sequence = (0..8)
        .map(|_| {
            first
                .sample(
                    &source,
                    SamplingStrategy::TopK {
                        temperature: 1.0,
                        k: 2,
                    },
                )
                .expect("top-k sample")
                .token_id
        })
        .collect::<Vec<_>>();
    let second_sequence = (0..8)
        .map(|_| {
            second
                .sample(
                    &source,
                    SamplingStrategy::TopK {
                        temperature: 1.0,
                        k: 2,
                    },
                )
                .expect("top-k sample")
                .token_id
        })
        .collect::<Vec<_>>();
    assert_eq!(first_sequence, second_sequence);
    assert!(first_sequence
        .iter()
        .all(|token_id| [1, 2].contains(token_id)));
}

#[test]
fn sampling_rejects_invalid_controls() {
    let source = logits(vec![1.0, 0.0]);
    assert!(temperature_distribution(&source, 0.0).is_err());
    assert!(top_k_distribution(&source, 1.0, 0).is_err());
    assert!(top_k_distribution(&source, 1.0, 3).is_err());
    assert!(top_p_distribution(&source, 1.0, 0.0).is_err());
    assert!(top_p_distribution(&source, 1.0, 1.1).is_err());
}
