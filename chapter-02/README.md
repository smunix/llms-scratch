# Chapter 2 — Working with Text Data

**Source:** Sebastian Raschka, *Build a Large Language Model (From Scratch)*, Chapter 2, pp. 17–49. This guide is an original Rust-oriented explanation of the chapter. It does **not** reproduce the book’s text or include the supplied source PDF. [1]

> **Chapter thesis.** A GPT-style model cannot consume characters or words directly. Before the transformer sees a sequence, text must travel through a deterministic preparation path: **text → tokens → token IDs → next-token input/target examples → token embeddings + positional embeddings**. [1]

## Learning objectives

Chapter 2 is the first implementation stage of the book’s end-to-end LLM workflow: data preparation and sampling. It establishes the representation and supervision signal that later chapters use for attention, model construction, and training. The key point is that an LLM’s first prediction problem is not “understand a sentence” in the abstract; it receives a fixed-length sequence of integer IDs and learns to predict the next ID at every position. [1]

| Section | Core idea | Rust counterpart in this folder |
|---|---|---|
| 2.1 | Discrete text must become continuous vectors. | `EmbeddingTable` in `src/lib.rs` |
| 2.2 | Tokenization splits text into processable units. | `simple_tokenize` and `simple_tokenizer.rs` |
| 2.3 | A vocabulary maps each token to one integer ID and supports reversal. | `build_vocab`, `SimpleTokenizer` |
| 2.4 | Special tokens represent unknown items and document boundaries. | `UNK`, `END_OF_TEXT`, fallback tokenizer |
| 2.5 | BPE uses subwords so unfamiliar spellings remain representable. | `toy_bpe_merges.rs` |
| 2.6 | A sliding window turns one token stream into shifted prediction pairs. | `sliding_window.rs` |
| 2.7 | An embedding table looks up a learnable vector for each token ID. | `EmbeddingTable::lookup` |
| 2.8 | Position vectors add order information missing from token lookup alone. | `embeddings_and_positions.rs` |

## 1. Why embeddings are necessary (Section 2.1)

Neural networks operate on numerical tensors, while raw words are categorical symbols. An **embedding** maps each discrete object to a dense vector of real values so that the network can perform matrix operations and gradients can update its representation. The same general idea exists for text, image, audio, and video, but an embedding model is tied to the input modality. Chapter 2 focuses on token-level embeddings because GPT-like models generate text one token at a time. [1]

The crucial distinction is between a pretrained, fixed embedding model such as Word2Vec and the input embedding table inside a language model. In a GPT-style model, token vectors are parameters of the model. They normally start from random values and are optimized together with all other parameters during next-token training, so their geometry becomes specific to the training distribution and objective. [1]

| Term | Meaning | Important implication |
|---|---|---|
| **Vocabulary size** `V` | Number of legal token IDs. | An embedding table has `V` rows. |
| **Embedding dimension** `d` | Width of each continuous representation. | Every token and position vector must have width `d`. |
| **Token embedding table** `E_tok ∈ ℝ^(V×d)` | Trainable mapping from IDs to vectors. | Token ID `k` selects row `E_tok[k]`. |
| **Hidden size** | Another common name for `d` in GPT configurations. | It affects model capacity and memory cost. |

The Rust examples intentionally implement a small vector table instead of a deep-learning framework. That makes the lookup operation visible: **an embedding layer is indexed row retrieval**, conceptually equivalent to multiplying a one-hot vector by the embedding matrix, but much more efficient than materializing a giant one-hot vector. [1]

## 2. Tokenization: decide what a “unit of text” is (Section 2.2)

A tokenizer turns a string into a sequence of tokens. The chapter begins with a transparent word-and-punctuation tokenizer. For a string such as `Hello, world. Is this-- a test?`, it separates word fragments from punctuation and removes whitespace for simplicity. Case is preserved: `Paris` and `paris` can mean different things to a model. [1]

Our `simple_tokenize` function makes this explicit. It recognizes whitespace and a small punctuation set as boundaries and returns punctuation as its own token. This is helpful for learning, but it is deliberately not a production tokenizer. It does not cover every Unicode boundary, it drops whitespace, and its rules are English-centric. Those limitations are useful reminders that tokenization is a modeling decision rather than mere string cleanup.

```rust
let tokens = simple_tokenize("Hello, world. Is This-- a test?");
assert_eq!(
    tokens,
    vec!["Hello", ",", "world", ".", "Is", "This", "--", "a", "test", "?"]
);
```

A simple tokenizer makes an important trade-off. Removing whitespace reduces the sequence length and simplifies the display, but whitespace may be semantically significant in some domains—for example, indentation-sensitive source code. GPT-style BPE tokenizers commonly represent leading spaces within their learned token units, rather than treating every space as an independent word delimiter. [1]

## 3. Vocabulary and reversible token IDs (Section 2.3)

A tokenizer alone emits strings; a neural network needs integers. A **vocabulary** is a one-to-one mapping between the unique token strings observed in the corpus and integer token IDs. For a reproducible classroom implementation, this project sorts the unique observed strings before enumerating them. The reverse map lets us decode model output back to text. The exact numeric IDs have no inherent linguistic meaning: the learned embedding table gives them meaning later. [1]

```rust
let training_tokens = simple_tokenize("Hello, world. This is a corpus.");
let vocab = build_vocab(&training_tokens, &[]);
let tokenizer = SimpleTokenizer::strict(vocab);

let ids = tokenizer.encode("Hello, world.")?;
let reconstructed = tokenizer.decode(&ids)?;
assert_eq!(reconstructed, "Hello, world.");
```

A strict word-level vocabulary has an immediate failure mode. If the corpus did not contain `Rustacean`, then encoding `Rustacean` cannot find an ID and should fail. This is called an **out-of-vocabulary (OOV)** problem. The failure is not a Rust error; it is a fundamental consequence of modeling tokens as a closed word list. [1]

## 4. Special context tokens (Section 2.4)

The chapter then extends the vocabulary with reserved markers. In the didactic tokenizer, `<|unk|>` replaces an unknown token, and `<|endoftext|>` marks the boundary between unrelated documents concatenated into one training stream. Without a document boundary marker, a sampler could create a training sequence whose prefix belongs to one article and whose target belongs to an unrelated article. [1]

```rust
let vocab = build_vocab(&training_tokens, &[END_OF_TEXT, UNK]);
let tokenizer = SimpleTokenizer::with_unknown(vocab, UNK)?;
let text = format!("Hello, Rustacean! {END_OF_TEXT} This is a corpus.");
let ids = tokenizer.encode(&text)?;
println!("{}", tokenizer.decode(&ids)?);
```

The book also distinguishes common sequence-control roles. `[BOS]` denotes a beginning, `[EOS]` marks an end, and `[PAD]` extends short examples so a batch has common length. GPT’s setup can use `<|endoftext|>` as an end or separator marker, and padded positions are typically ignored using an attention mask in later processing. The particular padding ID matters less when the loss and attention operations mask it consistently. [1]

| Token role | Purpose | This project |
|---|---|---|
| `<|unk|>` | Substitute an unavailable word-level token. | Implemented to teach the OOV case. |
| `<|endoftext|>` | Separate independent documents in a continuous token stream. | Implemented as `END_OF_TEXT`. |
| `[BOS]` | Mark a sequence start. | Discussed, not added to the toy vocabulary. |
| `[EOS]` | Mark a sequence end. | Conceptually similar to the document boundary token. |
| `[PAD]` | Equalize batch lengths. | Discussed only; batching is outside this small project. |

## 5. Byte-pair encoding (Section 2.5)

Word-only tokenization has a vocabulary-coverage problem: every unseen word becomes `<|unk|>`, which discards its spelling and any useful internal structure. **Byte-pair encoding (BPE)** addresses this by representing common strings as learned subword units while retaining a fallback to smaller units. It starts from small symbols, repeatedly merges frequently co-occurring adjacent pieces, and stores the resulting learned vocabulary and merge order. As a result, an unfamiliar string can normally be encoded as several known subwords or byte/character-level pieces instead of being replaced by a single unknown marker. [1]

The `toy_bpe_merges.rs` program demonstrates **application** of a hypothetical merge list, not training a full byte-level GPT tokenizer. Starting with `l | o | w | e | r`, merges such as `(l, o)`, `(lo, w)`, and `(e, r)` produce `low | er`. A production GPT-compatible BPE tokenizer requires a learned byte vocabulary, a large ranked merge table, Unicode/byte conversion rules, and special-token policy. The toy program is intentionally smaller so the causal idea of merging can be inspected directly.

```rust
let symbols = vec!["l", "o", "w", "e", "r"]
    .into_iter()
    .map(str::to_owned)
    .collect();
let result = apply_bpe_merges(symbols, &[("l", "o"), ("lo", "w"), ("e", "r")]);
assert_eq!(result, vec!["low", "er"]);
```

> **Practical distinction:** Use the simple tokenizer to learn the pipeline. Use a tested tokenizer implementation and the exact vocabulary/merge files that match a pretrained model when compatibility matters. Token IDs are only meaningful relative to the tokenizer and embedding matrix that were trained together.

## 6. Turn one long token stream into supervision (Section 2.6)

After tokenization, pretraining examples come from a single sequence of token IDs. Given IDs `z₀, z₁, …`, choose a context length `L`. An input window beginning at offset `i` is `xᵢ = [zᵢ, …, zᵢ₊ₗ₋₁]`. Its target row is shifted exactly one position: `yᵢ = [zᵢ₊₁, …, zᵢ₊ₗ]`. Training later asks the model at each input position to predict the corresponding entry in the target row. [1]

| Position | Input token ID | Target token ID |
|---:|---:|---:|
| 0 | `zᵢ` | `zᵢ₊₁` |
| 1 | `zᵢ₊₁` | `zᵢ₊₂` |
| … | … | … |
| `L - 1` | `zᵢ₊ₗ₋₁` | `zᵢ₊ₗ` |

The **stride** controls where the next example starts. With `stride = 1`, adjacent examples overlap heavily and the input window advances one token. With `stride = L`, input windows are adjacent and do not overlap. The chapter emphasizes that the context length, stride, and batch size have practical effects on computation, dataset reuse, and training behavior; they are hyperparameters rather than arbitrary details. [1]

```rust
let stream = vec![40, 367, 2885, 1464, 1807, 3619];
let examples = sliding_window_examples(&stream, 4, 1)?;
assert_eq!(examples[0].input_ids,  vec![40, 367, 2885, 1464]);
assert_eq!(examples[0].target_ids, vec![367, 2885, 1464, 1807]);
```

The Rust implementation returns `Vec<TrainingExample>` rather than a framework-specific tensor loader. That preserves the chapter’s data contract in a small, inspectable form. A training system would next batch these examples, optionally shuffle them, and discard or pad a partial final batch according to its training policy.

## 7. Token embedding lookup (Section 2.7)

Once each training example contains integer IDs, an embedding table converts every ID to a continuous vector. Let `E_tok` have shape `V × d`, where `V` is the vocabulary size and `d` is the embedding width. For `input_ids = [2, 3, 5, 1]`, the output is the stack of rows `E_tok[2]`, `E_tok[3]`, `E_tok[5]`, and `E_tok[1]`, with shape `4 × d`. For a batch of `B` examples each of length `L`, the shape becomes `B × L × d`. [1]

```rust
let table = EmbeddingTable::seeded(8, 3, 123);
let vectors = table.lookup(&[2, 3, 5, 1])?;
assert_eq!(vectors.len(), 4);       // sequence length
assert_eq!(vectors[0].len(), 3);    // embedding dimension
```

The table in this project is seeded only to make the printed values repeatable. In an actual language model, the entries are trainable floating-point parameters. Backpropagation adjusts them in response to next-token loss, so a token’s vector becomes useful in the context of the entire model rather than because the initial random values are intrinsically meaningful. [1]

## 8. Positional embeddings supply token order (Section 2.8)

A token lookup is position-independent: every occurrence of the same token ID receives the same token vector. Yet the sequence `dog bites man` has a different order and likely a different meaning from `man bites dog`. Transformer self-attention alone does not make absolute token position available in the initial input representation, so GPT-style models inject position information. [1]

The chapter contrasts two approaches. **Absolute positional embeddings** assign one vector to each valid position, such as position 0, 1, 2, and so on. **Relative positional embeddings** emphasize distances between positions, which can generalize naturally across some sequence lengths. The GPT configuration described in the chapter uses learned absolute position vectors. [1]

For token position `j`, the input to the model is:

```text
input_embedding[j] = token_embedding[token_id[j]] + position_embedding[j]
```

Both vectors have the same width `d`, so addition is elementwise. With a batch shape `B × L × d`, a position matrix of shape `L × d` is shared across batch members and added to each one. This project shows the one-sequence version explicitly:

```rust
let token_embeddings = token_table.lookup(&[2, 3, 5, 1])?;
let position_embeddings = position_table.lookup(&[0, 1, 2, 3])?;
let input_embeddings = add_absolute_positions(&token_embeddings, &position_embeddings)?;
```

The resulting matrix is the final Chapter 2 artifact: a position-aware, continuous representation ready for the transformer layers in the next chapter. The model must still enforce causal visibility during training and generation; Chapter 3 introduces the attention mechanisms that make this possible. [1]

## End-to-end mental model

| Step | Input | Output | Why it exists |
|---:|---|---|---|
| 1 | Raw documents | Text stream with boundary markers | Preserve corpus segmentation. |
| 2 | Text | Tokens | Decide the atomic symbols to model. |
| 3 | Tokens | Token IDs | Index a finite vocabulary efficiently. |
| 4 | Token-ID stream | `input_ids`, `target_ids` windows | Define next-token prediction supervision. |
| 5 | `input_ids` | Token vectors | Convert categorical IDs to continuous inputs. |
| 6 | Token vectors + positions | Input embeddings | Make order observable to the transformer. |

The chapter’s architecture is worth remembering because a bug at any stage changes the model’s learning problem. A tokenizer mismatch invalidates token IDs; a vocabulary mismatch retrieves the wrong embedding rows; an off-by-one target shift trains the wrong task; a missing document boundary creates artificial cross-document context; and incorrect positional shapes lose or scramble order information.

## Running the examples

From this directory, execute the following commands. No external model, network download, or source-book text is required.

```bash
cargo test
cargo run --bin simple_tokenizer
cargo run --bin toy_bpe_merges
cargo run --bin sliding_window
cargo run --bin embeddings_and_positions
```

The test suite verifies punctuation-aware tokenization, round-trip encoding/decoding, unknown-token substitution, shifted targets, embedding lookup, elementwise position addition, and ordered toy BPE merges.

## Limitations and sensible next steps

These programs deliberately optimize for inspectability. They do not implement production Unicode handling, byte-level BPE training, a high-performance tensor backend, batching, gradients, causal masking, or optimization. Those omissions are intentional: Chapter 2 ends at building correctly shaped input embeddings. A natural next milestone is to preserve this input contract while implementing causal self-attention.

## References

[1] Sebastian Raschka, “Working with Text Data,” Chapter 2 in *Build a Large Language Model (From Scratch)*, pp. 17–49, Manning, 2025. [Official book page][1]

[1]: https://www.manning.com/books/build-a-large-language-model-from-scratch
