# Sampling Next Tokens from Transformer Logits

This guide extends the Chapter 4 transformer package from hidden-state computation to **next-token selection**. It covers temperature scaling, top-k sampling, nucleus (top-p) sampling, greedy decoding, and reproducible categorical draws. The implementations are original Rust code using Candle tensors for the public logit and probability interfaces. [1] [2]

> **Scope boundary.** `TransformerBlock` returns contextual hidden states with shape `B × T × d`. A complete GPT model subsequently projects the last hidden state into a vocabulary-logit vector of shape `V`. The sampling utilities begin at that logit vector; they intentionally do not assume a tokenizer or vocabulary head. [1]

## Sampling interface

The public API accepts a **rank-one `F32` Candle tensor** containing one vocabulary logit per token ID. `TokenSampler` turns that vector into a next-token selection according to `SamplingStrategy`.

| API | Inputs | Output | Purpose |
|---|---|---|---|
| `temperature_distribution` | Logits, positive temperature | `FilteredDistribution` | Full-vocabulary probability distribution. |
| `top_k_distribution` | Logits, temperature, `k` | `FilteredDistribution` | Distribution supported on exactly the largest `k` scaled logits. |
| `top_p_distribution` | Logits, temperature, `p` | `FilteredDistribution` | Distribution supported on the minimal highest-probability prefix reaching mass `p`. |
| `TokenSampler::sample` | Logits, `SamplingStrategy` | `SampledToken` | Greedy or seeded categorical selection. |

`FilteredDistribution` retains the complete vocabulary-sized probability tensor; removed tokens receive probability zero. Its `retained_token_ids` list records the surviving support in descending-logit order. This makes filtering behavior inspectable without losing the original token-ID coordinate system.

## Temperature scaling

A model emits arbitrary real-valued logits rather than probabilities. Before stochastic sampling, the sampler divides every logit `zᵢ` by a positive temperature `τ` and applies a numerically stable softmax:

```text
pᵢ = exp(zᵢ / τ - max(z / τ)) / Σⱼ exp(zⱼ / τ - max(z / τ))
```

| Temperature choice | Effect on distribution | Practical interpretation |
|---|---|---|
| `0 < τ < 1` | More concentrated on high logits | Less random; sharper continuation choices. |
| `τ = 1` | Standard softmax | Uses the model’s original relative logit scale. |
| `τ > 1` | Flatter distribution | More random; lower-logit alternatives are more likely. |

`temperature_distribution` rejects zero, negative, NaN, and infinite temperatures. Lowering the temperature does **not** change the order of logits; it changes how strongly the categorical draw favors the largest ones. The test suite confirms that reducing temperature increases the probability of the largest logit and decreases tail probability. [2]

```rust
let logits = Tensor::from_vec(vec![2.0_f32, 1.4, 0.7, 0.1, -0.8], 5, &Device::Cpu)?;
let distribution = temperature_distribution(&logits, 0.7)?;
assert!((distribution.probabilities.to_vec1::<f32>()?.iter().sum::<f32>() - 1.0).abs() < 1e-6);
```

## Top-k sampling

Top-k sampling first applies temperature scaling, ranks token IDs by descending scaled logit, retains exactly `k` candidates, assigns every other token logit negative infinity, and applies softmax again. The result is a properly renormalized distribution over only those candidates.

```text
retained = k largest values in z / τ
p = softmax(masked(z / τ))
masked(zᵢ / τ) = zᵢ / τ      if i is retained
masked(zᵢ / τ) = -∞          otherwise
```

Top-k gives a fixed-size candidate set. It is straightforward to reason about and guarantees that no token outside the `k` largest scaled logits can be sampled. Its limitation is that one value of `k` may be too restrictive for uncertain contexts and unnecessarily broad for confident contexts. [2]

```rust
let distribution = top_k_distribution(&logits, 0.7, 3)?;
assert_eq!(distribution.retained_token_ids.len(), 3);
```

## Nucleus (top-p) sampling

Nucleus sampling, also called top-p sampling, adapts the support size to the model’s confidence. It first computes the full temperature-scaled distribution, sorts IDs by descending probability, and retains the **smallest prefix** whose cumulative probability reaches or exceeds `p`. At least the highest-probability token is kept for every valid `p > 0`. The remaining candidate logits are then renormalized through softmax. [2]

```text
ranked IDs: i₁, i₂, … such that p(i₁) ≥ p(i₂) ≥ …
retain smallest m such that Σᵣ₌₁ᵐ p(iᵣ) ≥ p
```

| Top-p choice | Support behavior |
|---|---|
| Small `p`, such as `0.5` | Often retains only the strongest candidates. |
| Moderate `p`, such as `0.8–0.95` | Adapts to the context’s uncertainty. |
| `p = 1` | Retains the full vocabulary, equivalent to temperature-only support. |

The implementation rejects `p ≤ 0` and `p > 1`. Compared with top-k, top-p does not impose a fixed candidate count; its retained set grows for ambiguous distributions and shrinks for sharply peaked ones. [2]

```rust
let distribution = top_p_distribution(&logits, 0.7, 0.85)?;
println!("Retained token IDs: {:?}", distribution.retained_token_ids);
```

## Greedy and categorical selection

`SamplingStrategy::Greedy` selects the largest raw logit and returns a one-hot probability tensor. It is deterministic and ignores temperature because no random draw occurs. The three stochastic strategies create a distribution and use `TokenSampler` for categorical selection.

```rust
let mut sampler = TokenSampler::seeded(123);
let next = sampler.sample(
    &logits,
    SamplingStrategy::TopP {
        temperature: 0.7,
        p: 0.85,
    },
)?;
println!("Next token ID: {}", next.token_id);
```

The sampler has a compact local pseudo-random-number generator. A fixed seed produces the same sequence of categorical draws, which is useful for demonstrations and tests. Production systems may substitute a cryptographically unrelated, platform-standard random source while preserving the same filtered probability distribution.

## Integrating with a GPT generation loop

A full generation loop repeatedly runs the model on its current token context, takes the **last-position vocabulary logits**, samples one token ID, appends it to the context, and repeats until a stopping rule is met. The present package provides the final distribution-and-selection stage but intentionally stops short of embedding lookup, vocabulary projection, cache management, tokenizer decoding, and context-window truncation.

```text
context IDs
  → token + position embeddings
  → stacked transformer blocks
  → final LayerNorm
  → vocabulary projection
  → last-position logits (V)
  → temperature / top-k / top-p filter
  → sampled token ID
  → append to context
```

This separation is deliberate: it lets sampling policies be tested against compact synthetic logits before they are attached to a trained vocabulary head.

## Validation coverage

| Test | Behavior verified |
|---|---|
| `lower_temperature_concentrates_probability_on_the_largest_logit` | Lower temperatures sharpen the full softmax distribution. |
| `top_k_keeps_exactly_the_highest_k_logits_and_renormalizes` | Removed tokens have zero mass, retained support has size `k`, and probabilities sum to one. |
| `top_p_keeps_the_smallest_descending_probability_prefix_reaching_threshold` | Nucleus support reaches `p` with the smallest ranked prefix. |
| `greedy_and_seeded_top_k_sampling_select_only_permitted_token_ids` | Greedy selects argmax; seeded categorical draws are reproducible and never leave top-k support. |
| `sampling_rejects_invalid_controls` | Invalid temperature, `k`, and `p` values return errors. |

## Running the demonstration

```bash
cargo test --all-targets
cargo run --bin sampling_strategies
```

The executable prints one fixed vocabulary-logit vector, the retained IDs and full probability tensors for temperature, top-k, and top-p filters, and the token IDs selected by greedy and seeded stochastic policies.

## References

[1] Sebastian Raschka, “Implementing a GPT Model from Scratch to Generate Text,” Chapter 4 in *Build a Large Language Model (From Scratch)*, Manning, 2025. [Official book page][1]

[2] Ari Holtzman et al., “The Curious Case of Neural Text Degeneration,” 2019. [Paper][2]

[3] Candle Core 0.6 API documentation. [Tensor API][3]

[1]: https://www.manning.com/books/build-a-large-language-model-from-scratch
[2]: https://arxiv.org/abs/1904.09751
[3]: https://docs.rs/candle-core/0.6.0/candle_core/
