use chapter_02_text_data::{
    add_absolute_positions, apply_bpe_merges, build_vocab, simple_tokenize, sliding_window_examples,
    EmbeddingTable, SimpleTokenizer, END_OF_TEXT, UNK,
};
use pretty_assertions::assert_eq;

#[test]
fn tokenizer_splits_words_and_punctuation_without_lowercasing() {
    assert_eq!(
        simple_tokenize("Hello, world. Is This-- a test?"),
        vec!["Hello", ",", "world", ".", "Is", "This", "--", "a", "test", "?"]
    );
}

#[test]
fn tokenizer_round_trips_known_text_and_substitutes_unknown_text() {
    let tokens = simple_tokenize("Hello, world.");
    let vocab = build_vocab(&tokens, &[END_OF_TEXT, UNK]);
    let tokenizer = SimpleTokenizer::with_unknown(vocab, UNK).expect("special token is present");

    let known_ids = tokenizer.encode("Hello, world.").expect("all tokens are known");
    assert_eq!(tokenizer.decode(&known_ids).expect("valid IDs"), "Hello, world.");

    let unseen_ids = tokenizer.encode("Hello, Rustacean!").expect("unknown token is substituted");
    assert_eq!(tokenizer.decode(&unseen_ids).expect("valid IDs"), "Hello, <|unk|> <|unk|>");
}

#[test]
fn sliding_window_creates_one_position_shifted_targets() {
    let examples = sliding_window_examples(&[10, 11, 12, 13, 14, 15], 3, 1).expect("valid configuration");
    assert_eq!(examples.len(), 3);
    assert_eq!(examples[0].input_ids, vec![10, 11, 12]);
    assert_eq!(examples[0].target_ids, vec![11, 12, 13]);
    assert_eq!(examples[2].input_ids, vec![12, 13, 14]);
    assert_eq!(examples[2].target_ids, vec![13, 14, 15]);
}

#[test]
fn embedding_lookup_retrieves_rows_and_position_vectors_are_added_elementwise() {
    let table = EmbeddingTable::seeded(6, 3, 123);
    let first_lookup = table.lookup(&[3]).expect("ID is in range");
    let repeated_lookup = table.lookup(&[3, 3]).expect("IDs are in range");
    assert_eq!(first_lookup[0], repeated_lookup[0]);

    let combined = add_absolute_positions(
        &[vec![1.0, 2.0], vec![3.0, 4.0]],
        &[vec![0.5, 0.5], vec![-1.0, 1.0]],
    )
    .expect("matching shapes");
    assert_eq!(combined, vec![vec![1.5, 2.5], vec![2.0, 5.0]]);
}

#[test]
fn bpe_merge_application_is_ordered() {
    let symbols = vec!["l", "o", "w", "e", "r"]
        .into_iter()
        .map(str::to_owned)
        .collect();
    assert_eq!(
        apply_bpe_merges(symbols, &[("l", "o"), ("lo", "w"), ("e", "r")]),
        vec!["low", "er"]
    );
}
