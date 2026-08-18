# Rust Example Walkthroughs

This guide explains the runnable programs in `src/bin`. Each program corresponds to a single transformation in the Chapter 2 input pipeline. The shared implementation is in `src/lib.rs`, and the assertions are in `tests/pipeline_tests.rs`.

## Design choices

The code is intentionally **framework-free**. A real training program would use a numerical/tensor library for batched GPU operations, automatic differentiation, and efficient tokenization. Here, ordinary Rust collections make the data shape and each transformation visible. The code favors iterator chains and typed `Result` values so failure cases—especially invalid IDs and OOV tokens—are part of the API rather than hidden side effects.

| Program | Demonstrates | Main output to inspect |
|---|---|---|
| `simple_tokenizer` | Token boundaries, vocabulary IDs, strict OOV failure, and `<|unk|>` fallback. | Token list, ID list, reconstructed text. |
| `toy_bpe_merges` | Applying a learned sequence of subword merges. | Initial symbols and merged subwords. |
| `sliding_window` | Next-token targets and the effect of stride. | Each `input -> targets` row. |
| `embeddings_and_positions` | Token lookup plus learned absolute position vectors. | The three matrices and final shape. |

## `simple_tokenizer`

### What the code does

`simple_tokenize` scans a string with one regular expression. It treats whitespace and a limited punctuation set as boundaries, returns punctuation as a token, and excludes whitespace from the result. The implementation preserves capitalization; it does not normalize words before lookup.

`build_vocab` deduplicates the observed tokens with `BTreeSet`, which provides a deterministic lexical order, then enumerates them into IDs. It appends special tokens after the observed vocabulary. `SimpleTokenizer` stores both directions of that mapping, so `encode` and `decode` are inverse operations for known text.

### Strict versus fallback encoding

`SimpleTokenizer::strict` has no unknown-token ID. It returns an error when it sees an OOV token. This is the correct behavior for exposing the limitation of a closed, word-level vocabulary.

`SimpleTokenizer::with_unknown` checks at construction time that its unknown marker is actually present in the vocabulary. During encoding, it replaces any token without a vocabulary entry with the configured marker’s ID. That behavior is simple and safe but lossy: several distinct unseen words collapse to the same vector. The BPE example shows why modern GPT-family tokenizers prefer subword decomposition instead.

### Key interface

```rust
let vocab = build_vocab(&tokens, &[END_OF_TEXT, UNK]);
let tokenizer = SimpleTokenizer::with_unknown(vocab, UNK)?;
let ids = tokenizer.encode("Hello, Rustacean!")?;
let text = tokenizer.decode(&ids)?;
```

The `?` operator makes failure propagation explicit. `encode` can fail with a strict vocabulary, and `decode` can fail if a caller gives an ID that is not in the inverse map.

## `toy_bpe_merges`

### What the code does

BPE learning observes token frequencies to decide which adjacent symbols should be merged. The code in this folder does **not** learn those frequencies. Instead, it receives an already learned ordered merge list and applies it to a small sequence. This isolates the most important mechanical insight: a later merge can rely on the result of an earlier merge.

```rust
let result = apply_bpe_merges(
    vec!["l", "o", "w", "e", "r"].into_iter().map(str::to_owned).collect(),
    &[("l", "o"), ("lo", "w"), ("e", "r")],
);
assert_eq!(result, vec!["low", "er"]);
```

`apply_bpe_merges` walks through each merge rank in sequence. Within one pass, its `fold` holds the already emitted symbols. When the previous emitted symbol and current symbol match the selected pair, it pops the former and pushes their concatenation. Otherwise, it keeps the current symbol unchanged.

### What it intentionally does not claim

This is not a compatible implementation of a GPT tokenizer. A production implementation must operate on bytes, correctly handle Unicode and leading spaces, manage a large learned vocabulary and merge-ranking table, and define special-token behavior. Use a mature tokenizer that matches the model checkpoint in real projects.

## `sliding_window`

### What the code does

A next-token model is supervised by a one-position shift. The helper takes an entire token stream plus two hyperparameters and returns `TrainingExample` values.

```rust
let examples = sliding_window_examples(&token_ids, context_length, stride)?;
```

For a start offset `i`, it creates two slices:

```text
input_ids  = token_ids[i .. i + context_length]
target_ids = token_ids[i + 1 .. i + context_length + 1]
```

The function rejects `context_length = 0` and `stride = 0`, because both would make the sampling contract invalid. It uses `saturating_sub` before producing start positions so a short input stream simply yields no complete examples rather than attempting an invalid slice.

### Why stride matters

| Configuration | First input | Second input | Interpretation |
|---|---|---|---|
| `context_length = 4`, `stride = 1` | IDs `0..4` | IDs `1..5` | High overlap; dense reuse of tokens. |
| `context_length = 4`, `stride = 4` | IDs `0..4` | IDs `4..8` | Adjacent contexts; no input overlap. |

Each row carries a full target vector because a causal language model usually makes one prediction per input position in parallel during training. It still may only attend to the current and earlier positions; the attention mask that enforces that constraint belongs to the next architectural stage.

## `embeddings_and_positions`

### What the code does

`EmbeddingTable::seeded(vocab_size, embedding_dim, seed)` constructs a small matrix of repeatable floating-point values. The pseudo-random generator is for a stable demo only. `lookup` then retrieves one matrix row for each token ID.

```rust
let token_embeddings = token_table.lookup(&[2, 3, 5, 1])?;
```

If the table has shape `V × d` and the input has `L` IDs, the result has shape `L × d`. The function checks all IDs and returns a descriptive error instead of silently returning a wrong row.

The same lookup mechanism supplies positional vectors. We use position IDs `[0, 1, 2, …]`, then call `add_absolute_positions` to add corresponding dimensions row by row.

```rust
let input_embeddings = add_absolute_positions(
    &token_embeddings,
    &position_embeddings,
)?;
```

The function validates that sequence lengths match and that every token vector and position vector have the same width. These checks make a common tensor-shape bug explicit before later model layers receive malformed inputs.

## Test suite as executable explanation

`cargo test` validates the following pipeline invariants.

| Test | Invariant checked |
|---|---|
| `tokenizer_splits_words_and_punctuation_without_lowercasing` | Word boundaries and case preservation are predictable. |
| `tokenizer_round_trips_known_text_and_substitutes_unknown_text` | Known tokens decode correctly and OOV tokens use the fallback policy. |
| `sliding_window_creates_one_position_shifted_targets` | Every target row is its input shifted by exactly one position. |
| `embedding_lookup_retrieves_rows_and_position_vectors_are_added_elementwise` | The same ID retrieves the same vector, and position addition is elementwise. |
| `bpe_merge_application_is_ordered` | Earlier BPE merges can create symbols used by later merges. |

## Suggested experiments

Change only one variable at a time and compare the printed data. Add a punctuation character to `simple_tokenize`; then observe the new vocabulary. Run `sliding_window` with a stride larger than the context length and note the skipped IDs. Change the two embedding dimensions in `embeddings_and_positions` and confirm that position addition still needs matching widths. Finally, reverse the order of two BPE merges and observe that the resulting subwords can change. These experiments connect implementation details to the chapter’s core theme: data representation determines what the model can learn from.
