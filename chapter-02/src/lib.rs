//! Reusable, deliberately small building blocks for the Chapter 2 examples.
//!
//! These examples explain the data path before the transformer: text -> tokens ->
//! token IDs -> shifted training pairs -> token embeddings + positional embeddings.
//! They are pedagogical implementations, not production tokenizers or tensor kernels.

use itertools::Itertools;
use regex::Regex;
use std::collections::{BTreeSet, HashMap};

pub const UNK: &str = "<|unk|>";
pub const END_OF_TEXT: &str = "<|endoftext|>";

/// Split prose into words plus a small collection of punctuation tokens.
///
/// Whitespace is treated as a separator rather than as a token. The function preserves
/// letter case because case can carry useful information for language modeling.
pub fn simple_tokenize(text: &str) -> Vec<String> {
    let boundary = Regex::new(r#"--|[,.:;?_!\"()']|\s+"#).expect("tokenizer regex is valid");
    let (mut tokens, cursor) = boundary.find_iter(text).fold(
        (Vec::new(), 0_usize),
        |(mut out, cursor), matched| {
            let word = text[cursor..matched.start()].trim();
            (!word.is_empty()).then(|| out.push(word.to_owned()));

            let separator = matched.as_str();
            (!separator.trim().is_empty()).then(|| out.push(separator.to_owned()));
            (out, matched.end())
        },
    );

    let tail = text[cursor..].trim();
    (!tail.is_empty()).then(|| tokens.push(tail.to_owned()));
    tokens
}

/// Build a deterministic vocabulary from observed tokens, then append requested special tokens.
pub fn build_vocab(tokens: &[String], special_tokens: &[&str]) -> HashMap<String, usize> {
    let ordered = tokens
        .iter()
        .cloned()
        .collect::<BTreeSet<_>>()
        .into_iter()
        .chain(special_tokens.iter().map(|token| (*token).to_owned()))
        .unique()
        .collect_vec();

    ordered
        .into_iter()
        .enumerate()
        .map(|(id, token)| (token, id))
        .collect()
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SimpleTokenizer {
    str_to_id: HashMap<String, usize>,
    id_to_str: HashMap<usize, String>,
    unknown_token: Option<String>,
}

impl SimpleTokenizer {
    /// Construct a strict tokenizer. Unknown tokens produce an error during encoding.
    pub fn strict(vocab: HashMap<String, usize>) -> Self {
        Self::new(vocab, None)
    }

    /// Construct a tokenizer that replaces out-of-vocabulary tokens with `unknown_token`.
    pub fn with_unknown(vocab: HashMap<String, usize>, unknown_token: &str) -> Result<Self, String> {
        (!vocab.contains_key(unknown_token))
            .then(|| format!("unknown token {unknown_token:?} is absent from the vocabulary"))
            .map_or_else(
                || Ok(Self::new(vocab, Some(unknown_token.to_owned()))),
                Err,
            )
    }

    fn new(vocab: HashMap<String, usize>, unknown_token: Option<String>) -> Self {
        let id_to_str = vocab
            .iter()
            .map(|(token, id)| (*id, token.clone()))
            .collect();
        Self {
            str_to_id: vocab,
            id_to_str,
            unknown_token,
        }
    }

    pub fn vocab_size(&self) -> usize {
        self.str_to_id.len()
    }

    /// Convert text into token IDs. A strict tokenizer returns an explanatory error for an OOV token.
    pub fn encode(&self, text: &str) -> Result<Vec<usize>, String> {
        simple_tokenize(text)
            .into_iter()
            .map(|token| {
                self.str_to_id
                    .get(&token)
                    .copied()
                    .or_else(|| {
                        self.unknown_token
                            .as_ref()
                            .and_then(|unk| self.str_to_id.get(unk).copied())
                    })
                    .ok_or_else(|| format!("out-of-vocabulary token: {token:?}"))
            })
            .collect()
    }

    /// Convert token IDs to readable text and remove the spaces inserted before punctuation.
    pub fn decode(&self, ids: &[usize]) -> Result<String, String> {
        let joined = ids
            .iter()
            .map(|id| {
                self.id_to_str
                    .get(id)
                    .cloned()
                    .ok_or_else(|| format!("unknown token ID: {id}"))
            })
            .collect::<Result<Vec<_>, _>>()?
            .into_iter()
            .join(" ");
        let before_punctuation = Regex::new(r#"\s+([,.:;?!\"()'])"#).expect("decode regex is valid");
        Ok(before_punctuation.replace_all(&joined, "$1").into_owned())
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TrainingExample {
    pub input_ids: Vec<usize>,
    pub target_ids: Vec<usize>,
}

/// Build next-token training examples with a sliding input window.
///
/// For a window starting at `i`, the target sequence is the input sequence shifted right
/// by one position. `stride = 1` makes highly overlapping examples; a stride equal to
/// `context_length` produces adjacent, non-overlapping inputs.
pub fn sliding_window_examples(
    token_ids: &[usize],
    context_length: usize,
    stride: usize,
) -> Result<Vec<TrainingExample>, String> {
    if context_length == 0 {
        return Err("context_length must be greater than zero".to_owned());
    }
    if stride == 0 {
        return Err("stride must be greater than zero".to_owned());
    }

    let upper_bound = token_ids.len().saturating_sub(context_length);
    Ok((0..upper_bound)
        .step_by(stride)
        .map(|start| TrainingExample {
            input_ids: token_ids[start..start + context_length].to_vec(),
            target_ids: token_ids[start + 1..start + context_length + 1].to_vec(),
        })
        .collect())
}

/// A tiny embedding table. A real model learns these weights with backpropagation.
#[derive(Debug, Clone)]
pub struct EmbeddingTable {
    weights: Vec<Vec<f32>>,
}

impl EmbeddingTable {
    /// Make deterministic pseudo-random vectors without relying on a machine-learning framework.
    pub fn seeded(vocab_size: usize, embedding_dim: usize, seed: u64) -> Self {
        let (_, weights) = (0..vocab_size).fold((seed, Vec::with_capacity(vocab_size)), |(state, mut rows), _| {
            let (next_state, row) = (0..embedding_dim).fold((state, Vec::with_capacity(embedding_dim)), |(state, mut row), _| {
                let next = state.wrapping_mul(6364136223846793005).wrapping_add(1);
                let unit = ((next >> 32) as f32) / (u32::MAX as f32);
                row.push(unit - 0.5);
                (next, row)
            });
            rows.push(row);
            (next_state, rows)
        });
        Self { weights }
    }

    pub fn rows(&self) -> usize {
        self.weights.len()
    }

    pub fn embedding_dim(&self) -> usize {
        self.weights.first().map_or(0, Vec::len)
    }

    /// Perform an embedding lookup: ID `k` retrieves row `k`.
    pub fn lookup(&self, token_ids: &[usize]) -> Result<Vec<Vec<f32>>, String> {
        token_ids
            .iter()
            .map(|id| {
                self.weights
                    .get(*id)
                    .cloned()
                    .ok_or_else(|| format!("token ID {id} exceeds vocabulary size {}", self.rows()))
            })
            .collect()
    }
}

/// Add a learned absolute position vector to every token vector in one sequence.
///
/// The two matrices must have the same sequence length and embedding width. This is the
/// operation usually broadcast across every batch member in a GPT-style input pipeline.
pub fn add_absolute_positions(
    token_embeddings: &[Vec<f32>],
    position_embeddings: &[Vec<f32>],
) -> Result<Vec<Vec<f32>>, String> {
    if token_embeddings.len() != position_embeddings.len() {
        return Err("token and position sequences must have the same length".to_owned());
    }

    token_embeddings
        .iter()
        .zip(position_embeddings)
        .enumerate()
        .map(|(position, (token, positional))| {
            if token.len() != positional.len() {
                return Err(format!("embedding width mismatch at position {position}"));
            }
            Ok(token.iter().zip(positional).map(|(a, b)| a + b).collect_vec())
        })
        .collect()
}

/// Apply a learned BPE merge list to an already-split sequence of symbol strings.
///
/// This demonstrates merge *application*, not byte-level GPT tokenization or BPE training.
pub fn apply_bpe_merges(mut symbols: Vec<String>, merges: &[(&str, &str)]) -> Vec<String> {
    for (left, right) in merges {
        symbols = symbols.into_iter().fold(Vec::new(), |mut merged, symbol| {
            let should_merge = merged.last().is_some_and(|previous| previous == left) && symbol == *right;
            if should_merge {
                let previous = merged.pop().expect("last element exists when merging");
                merged.push(format!("{previous}{symbol}"));
            } else {
                merged.push(symbol);
            }
            merged
        });
    }
    symbols
}
