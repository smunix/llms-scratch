//! Reusable, deliberately small building blocks for the Chapter 2 examples.
//!
//! These examples explain the data path before the transformer: text -> tokens ->
//! token IDs -> shifted training pairs -> token embeddings + positional embeddings.
//! They are pedagogical implementations, not production tokenizers or tensor kernels.

use candle_core::{Device, Tensor};
use fancy_regex::Regex as FancyRegex;
use itertools::Itertools;
use regex::Regex;
use std::collections::{BTreeSet, HashMap};
use std::fs;
use std::path::Path;

pub const UNK: &str = "<|unk|>";
pub const END_OF_TEXT: &str = "<|endoftext|>";

/// Split prose into words plus a small collection of punctuation tokens.
///
/// Whitespace is treated as a separator rather than as a token. The function preserves
/// letter case because case can carry useful information for language modeling.
pub fn simple_tokenize(text: &str) -> Vec<String> {
    let boundary = Regex::new(r#"--|[,.:;?_!\"()']|\s+"#).expect("tokenizer regex is valid");
    let (mut tokens, cursor) =
        boundary
            .find_iter(text)
            .fold((Vec::new(), 0_usize), |(mut out, cursor), matched| {
                let word = text[cursor..matched.start()].trim();
                (!word.is_empty()).then(|| out.push(word.to_owned()));

                let separator = matched.as_str();
                (!separator.trim().is_empty()).then(|| out.push(separator.to_owned()));
                (out, matched.end())
            });

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
    pub fn with_unknown(
        vocab: HashMap<String, usize>,
        unknown_token: &str,
    ) -> Result<Self, String> {
        (!vocab.contains_key(unknown_token))
            .then(|| format!("unknown token {unknown_token:?} is absent from the vocabulary"))
            .map_or_else(|| Ok(Self::new(vocab, Some(unknown_token.to_owned()))), Err)
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
        let before_punctuation =
            Regex::new(r#"\s+([,.:;?!\"()'])"#).expect("decode regex is valid");
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

/// A tiny Candle embedding table with shape `(vocabulary × embedding width)`.
/// A real model learns these weights with backpropagation.
#[derive(Debug, Clone)]
pub struct EmbeddingTable {
    weights: Tensor,
    vocab_size: usize,
    embedding_dim: usize,
}

impl EmbeddingTable {
    /// Make deterministic pseudo-random vectors directly as a Candle CPU tensor.
    pub fn seeded(vocab_size: usize, embedding_dim: usize, seed: u64) -> Self {
        let (_, values) = (0..(vocab_size * embedding_dim)).fold(
            (seed, Vec::with_capacity(vocab_size * embedding_dim)),
            |(state, mut values), _| {
                let next = state.wrapping_mul(6364136223846793005).wrapping_add(1);
                let unit = ((next >> 32) as f32) / (u32::MAX as f32);
                values.push(unit - 0.5);
                (next, values)
            },
        );
        let weights = Tensor::from_vec(values, (vocab_size, embedding_dim), &Device::Cpu)
            .expect("seeded embedding dimensions are valid");
        Self {
            weights,
            vocab_size,
            embedding_dim,
        }
    }

    pub fn rows(&self) -> usize {
        self.vocab_size
    }

    pub fn embedding_dim(&self) -> usize {
        self.embedding_dim
    }

    /// Expose the educational weight table as a Candle tensor.
    pub fn tensor(&self) -> &Tensor {
        &self.weights
    }

    /// Perform an embedding lookup: ID `k` retrieves row `k` through Candle `index_select`.
    pub fn lookup(&self, token_ids: &[usize]) -> Result<Tensor, String> {
        let indices = token_ids
            .iter()
            .map(|id| {
                (*id < self.rows())
                    .then(|| {
                        u32::try_from(*id).map_err(|_| format!("token ID {id} does not fit in u32"))
                    })
                    .ok_or_else(|| {
                        format!("token ID {id} exceeds vocabulary size {}", self.rows())
                    })?
            })
            .collect::<Result<Vec<_>, String>>()?;
        let index_tensor = Tensor::from_vec(indices, token_ids.len(), self.weights.device())
            .map_err(|error| format!("could not create Candle token-index tensor: {error}"))?;
        self.weights
            .index_select(&index_tensor, 0)
            .map_err(|error| format!("Candle embedding lookup failed: {error}"))
    }
}

/// Add learned absolute position vectors to token vectors with Candle elementwise addition.
///
/// The two tensors must have the same `(sequence length, embedding width)` shape. This is the
/// operation usually broadcast across every batch member in a GPT-style input pipeline.
pub fn add_absolute_positions(
    token_embeddings: &Tensor,
    position_embeddings: &Tensor,
) -> Result<Tensor, String> {
    let token_shape = token_embeddings
        .dims2()
        .map_err(|error| format!("token embeddings must be rank 2: {error}"))?;
    let position_shape = position_embeddings
        .dims2()
        .map_err(|error| format!("position embeddings must be rank 2: {error}"))?;
    (token_shape == position_shape)
        .then_some(())
        .ok_or_else(|| {
            format!(
                "embedding shapes must match: token={token_shape:?}, position={position_shape:?}"
            )
        })?;
    token_embeddings
        .broadcast_add(position_embeddings)
        .map_err(|error| format!("Candle positional addition failed: {error}"))
}

/// Build position-aware input embeddings with Candle tensors on CPU.
///
/// Candle performs row lookup with `index_select` and adds the position tensor to the token tensor.
/// The returned tensor has shape `(sequence length, embedding width)`; adding a leading batch axis
/// is a separate batching concern.
pub fn candle_input_embeddings(
    token_table: &EmbeddingTable,
    position_table: &EmbeddingTable,
    token_ids: &[usize],
) -> Result<Tensor, String> {
    if token_table.embedding_dim() != position_table.embedding_dim() {
        return Err("token and position embedding widths must match".to_owned());
    }
    if token_ids.len() > position_table.rows() {
        return Err(format!(
            "sequence length {} exceeds position-table length {}",
            token_ids.len(),
            position_table.rows()
        ));
    }

    let position_ids = (0..token_ids.len()).collect_vec();
    let token_embeddings = token_table.lookup(token_ids)?;
    let position_embeddings = position_table.lookup(&position_ids)?;
    add_absolute_positions(&token_embeddings, &position_embeddings)
}

/// Apply a learned BPE merge list to an already-split sequence of symbol strings.
///
/// This demonstrates merge *application*, not byte-level GPT tokenization or BPE training.
pub fn apply_bpe_merges(mut symbols: Vec<String>, merges: &[(&str, &str)]) -> Vec<String> {
    for (left, right) in merges {
        symbols = symbols.into_iter().fold(Vec::new(), |mut merged, symbol| {
            let should_merge =
                merged.last().is_some_and(|previous| previous == left) && symbol == *right;
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

/// Build GPT-2's reversible byte-to-Unicode mapping.
///
/// GPT-2 begins with all 256 byte values, but maps bytes that are inconvenient in ordinary
/// Unicode text (for example, the space byte) into unused Unicode code points. This lets BPE
/// operate over Unicode strings while preserving arbitrary UTF-8 bytes without replacement
/// characters or an unknown-token fallback.
pub fn gpt2_byte_to_unicode() -> (HashMap<u8, char>, HashMap<char, u8>) {
    let visible_bytes = (b'!'..=b'~')
        .chain(0xA1_u8..=0xAC_u8)
        .chain(0xAE_u8..=0xFF_u8)
        .collect_vec();

    let mut byte_values = visible_bytes.clone();
    let mut code_points = visible_bytes.iter().map(|byte| *byte as u32).collect_vec();
    let mut next_code_point = 256_u32;

    (0_u8..=255_u8)
        .filter(|byte| !visible_bytes.contains(byte))
        .for_each(|byte| {
            byte_values.push(byte);
            code_points.push(next_code_point);
            next_code_point += 1;
        });

    let byte_to_unicode = byte_values
        .into_iter()
        .zip(code_points)
        .map(|(byte, code_point)| {
            (
                byte,
                char::from_u32(code_point)
                    .expect("GPT-2 byte mapping uses valid Unicode code points"),
            )
        })
        .collect::<HashMap<_, _>>();
    let unicode_to_byte = byte_to_unicode
        .iter()
        .map(|(byte, character)| (*character, *byte))
        .collect::<HashMap<_, _>>();

    (byte_to_unicode, unicode_to_byte)
}

/// A byte-level BPE tokenizer compatible with released GPT-2 `encoder.json` and `vocab.bpe`
/// artifacts.
///
/// The tokenizer deliberately keeps its artifact loader public. A vocabulary, merge list, and
/// embedding table are a matched set: swapping any one of them changes the IDs a GPT model sees.
#[derive(Debug, Clone)]
pub struct Gpt2BpeTokenizer {
    encoder: HashMap<String, u32>,
    decoder: HashMap<u32, String>,
    merge_ranks: HashMap<(String, String), usize>,
    byte_encoder: HashMap<u8, char>,
    byte_decoder: HashMap<char, u8>,
    pretokenizer: FancyRegex,
    special_tokens: HashMap<String, u32>,
}

impl Gpt2BpeTokenizer {
    /// Load the standard GPT-2 vocabulary JSON and BPE merge file.
    ///
    /// `encoder_path` maps serialized BPE strings to IDs. `merges_path` begins with a version
    /// header; every later non-empty line is an ordered pair whose file order determines rank.
    pub fn from_files(
        encoder_path: impl AsRef<Path>,
        merges_path: impl AsRef<Path>,
    ) -> Result<Self, String> {
        let encoder_path = encoder_path.as_ref();
        let merges_path = merges_path.as_ref();
        let encoder_json = fs::read_to_string(encoder_path)
            .map_err(|error| format!("could not read {}: {error}", encoder_path.display()))?;
        let encoder = serde_json::from_str::<HashMap<String, u32>>(&encoder_json)
            .map_err(|error| format!("could not parse {}: {error}", encoder_path.display()))?;
        let decoder = encoder
            .iter()
            .map(|(token, id)| (*id, token.clone()))
            .collect::<HashMap<_, _>>();
        if decoder.len() != encoder.len() {
            return Err("encoder.json maps multiple token strings to the same token ID".to_owned());
        }

        let merges_text = fs::read_to_string(merges_path)
            .map_err(|error| format!("could not read {}: {error}", merges_path.display()))?;
        let merge_ranks = merges_text
            .lines()
            .skip(1)
            .filter(|line| !line.trim().is_empty())
            .enumerate()
            .map(|(rank, line)| {
                let pair = line.split_whitespace().collect_vec();
                (pair.len() == 2)
                    .then(|| ((pair[0].to_owned(), pair[1].to_owned()), rank))
                    .ok_or_else(|| format!("invalid BPE merge line: {line:?}"))
            })
            .collect::<Result<HashMap<_, _>, _>>()?;

        let (byte_encoder, byte_decoder) = gpt2_byte_to_unicode();
        let pretokenizer = FancyRegex::new(
            r"'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+",
        )
        .map_err(|error| format!("could not build GPT-2 pre-tokenizer: {error}"))?;
        let special_tokens = [END_OF_TEXT]
            .into_iter()
            .filter_map(|token| encoder.get(token).map(|id| (token.to_owned(), *id)))
            .collect();

        Ok(Self {
            encoder,
            decoder,
            merge_ranks,
            byte_encoder,
            byte_decoder,
            pretokenizer,
            special_tokens,
        })
    }

    pub fn vocabulary_size(&self) -> usize {
        self.encoder.len()
    }

    pub fn special_token_id(&self, token: &str) -> Option<u32> {
        self.special_tokens.get(token).copied()
    }

    pub fn token_for_id(&self, id: u32) -> Option<&str> {
        self.decoder.get(&id).map(String::as_str)
    }

    /// Return GPT-2's serialized BPE token strings before their conversion to integer IDs.
    pub fn tokenize(&self, text: &str) -> Result<Vec<String>, String> {
        text.split(END_OF_TEXT)
            .enumerate()
            .try_fold(Vec::new(), |mut pieces, (index, segment)| {
                (index > 0).then(|| pieces.push(END_OF_TEXT.to_owned()));
                pieces.extend(self.tokenize_without_special(segment)?);
                Ok(pieces)
            })
    }

    /// Convert text to GPT-2 vocabulary IDs. The method preserves `<|endoftext|>` as its special ID.
    pub fn encode(&self, text: &str) -> Result<Vec<u32>, String> {
        self.tokenize(text)?
            .into_iter()
            .map(|piece| {
                self.encoder.get(&piece).copied().ok_or_else(|| {
                    format!("BPE result is absent from the loaded vocabulary: {piece:?}")
                })
            })
            .collect()
    }

    /// Decode GPT-2 IDs back into UTF-8 text by reversing both the vocabulary lookup and byte map.
    pub fn decode(&self, ids: &[u32]) -> Result<String, String> {
        let byte_stream = ids
            .iter()
            .map(|id| {
                self.decoder
                    .get(id)
                    .ok_or_else(|| format!("unknown token ID: {id}"))
            })
            .collect::<Result<Vec<_>, _>>()?
            .iter()
            .flat_map(|token| token.chars())
            .map(|character| {
                self.byte_decoder.get(&character).copied().ok_or_else(|| {
                    format!("token character is not part of the GPT-2 byte map: {character:?}")
                })
            })
            .collect::<Result<Vec<_>, _>>()?;
        String::from_utf8(byte_stream)
            .map_err(|error| format!("decoded bytes are not valid UTF-8: {error}"))
    }

    fn tokenize_without_special(&self, text: &str) -> Result<Vec<String>, String> {
        self.pretokenizer
            .find_iter(text)
            .map(|matched| {
                let matched =
                    matched.map_err(|error| format!("GPT-2 pre-tokenization failed: {error}"))?;
                let byte_encoded = matched
                    .as_str()
                    .as_bytes()
                    .iter()
                    .map(|byte| {
                        self.byte_encoder
                            .get(byte)
                            .copied()
                            .ok_or_else(|| format!("missing byte mapping for {byte}"))
                    })
                    .collect::<Result<String, _>>()?;
                Ok(self.apply_ranked_bpe(&byte_encoded))
            })
            .collect::<Result<Vec<_>, String>>()
            .map(|groups| groups.into_iter().flatten().collect())
    }

    fn apply_ranked_bpe(&self, byte_encoded_piece: &str) -> Vec<String> {
        let mut symbols = byte_encoded_piece
            .chars()
            .map(|character| character.to_string())
            .collect_vec();

        while symbols.len() > 1 {
            let best_pair = symbols
                .windows(2)
                .filter_map(|window| {
                    let pair = (window[0].clone(), window[1].clone());
                    self.merge_ranks.get(&pair).map(|rank| (*rank, pair))
                })
                .min_by_key(|(rank, _)| *rank)
                .map(|(_, pair)| pair);

            let Some((left, right)) = best_pair else {
                break;
            };
            symbols = symbols.into_iter().fold(Vec::new(), |mut merged, symbol| {
                let should_merge =
                    merged.last().is_some_and(|previous| previous == &left) && symbol == right;
                if should_merge {
                    let previous = merged
                        .pop()
                        .expect("an element exists when BPE merges a pair");
                    merged.push(format!("{previous}{symbol}"));
                } else {
                    merged.push(symbol);
                }
                merged
            });
        }
        symbols
    }
}
