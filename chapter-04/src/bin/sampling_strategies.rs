use candle_core::{Device, Tensor};
use chapter_04_transformer::{
    temperature_distribution, top_k_distribution, top_p_distribution, SamplingStrategy,
    TokenSampler,
};

fn main() -> Result<(), String> {
    // A compact stand-in for the final vocabulary-logit vector of a GPT model.
    let logits = Tensor::from_vec(vec![2.0_f32, 1.4, 0.7, 0.1, -0.8], 5, &Device::Cpu)
        .map_err(|error| format!("could not create vocabulary logits: {error}"))?;
    let temperature = temperature_distribution(&logits, 0.7)?;
    let top_k = top_k_distribution(&logits, 0.7, 3)?;
    let top_p = top_p_distribution(&logits, 0.7, 0.85)?;
    let mut sampler = TokenSampler::seeded(123);

    println!(
        "Vocabulary logits: {:?}",
        logits.to_vec1::<f32>().map_err(|error| error.to_string())?
    );
    print_distribution("temperature = 0.7", &temperature)?;
    print_distribution("top-k = 3, temperature = 0.7", &top_k)?;
    print_distribution("top-p = 0.85, temperature = 0.7", &top_p)?;

    let greedy = sampler.sample(&logits, SamplingStrategy::Greedy)?;
    let sampled_top_k = sampler.sample(
        &logits,
        SamplingStrategy::TopK {
            temperature: 0.7,
            k: 3,
        },
    )?;
    let sampled_top_p = sampler.sample(
        &logits,
        SamplingStrategy::TopP {
            temperature: 0.7,
            p: 0.85,
        },
    )?;
    println!("Greedy selected token ID: {}", greedy.token_id);
    println!("Seeded top-k sampled token ID: {}", sampled_top_k.token_id);
    println!("Seeded top-p sampled token ID: {}", sampled_top_p.token_id);
    println!(
        "These utilities consume the final vocabulary logits emitted after a transformer stack and vocabulary projection."
    );
    Ok(())
}

fn print_distribution(
    name: &str,
    distribution: &chapter_04_transformer::FilteredDistribution,
) -> Result<(), String> {
    let probabilities = distribution
        .probabilities
        .to_vec1::<f32>()
        .map_err(|error| format!("could not materialize {name} probabilities: {error}"))?;
    println!("\n{name}");
    println!("  retained IDs: {:?}", distribution.retained_token_ids);
    println!("  probabilities: {probabilities:?}");
    Ok(())
}
