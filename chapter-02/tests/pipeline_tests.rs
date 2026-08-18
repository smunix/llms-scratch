use chapter_02_text_data::{
    add_absolute_positions, apply_bpe_merges, build_vocab, candle_input_embeddings,
    gpt2_byte_to_unicode, simple_tokenize, sliding_window_examples, EmbeddingTable,
    Gpt2BpeTokenizer, SimpleTokenizer, END_OF_TEXT, UNK,
};
use nalgebra::DMatrix;
use pretty_assertions::assert_eq;
use std::path::PathBuf;

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

    let known_ids = tokenizer
        .encode("Hello, world.")
        .expect("all tokens are known");
    assert_eq!(
        tokenizer.decode(&known_ids).expect("valid IDs"),
        "Hello, world."
    );

    let unseen_ids = tokenizer
        .encode("Hello, Rustacean!")
        .expect("unknown token is substituted");
    assert_eq!(
        tokenizer.decode(&unseen_ids).expect("valid IDs"),
        "Hello, <|unk|> <|unk|>"
    );
}

#[test]
fn sliding_window_creates_one_position_shifted_targets() {
    let examples =
        sliding_window_examples(&[10, 11, 12, 13, 14, 15], 3, 1).expect("valid configuration");
    assert_eq!(examples.len(), 3);
    assert_eq!(examples[0].input_ids, vec![10, 11, 12]);
    assert_eq!(examples[0].target_ids, vec![11, 12, 13]);
    assert_eq!(examples[2].input_ids, vec![12, 13, 14]);
    assert_eq!(examples[2].target_ids, vec![13, 14, 15]);
}

#[test]
fn nalgebra_embedding_lookup_retrieves_rows_and_position_vectors_are_added_elementwise() {
    let table = EmbeddingTable::seeded(6, 3, 123);
    let first_lookup = table.lookup(&[3]).expect("ID is in range");
    let repeated_lookup = table.lookup(&[3, 3]).expect("IDs are in range");
    assert_eq!(first_lookup.nrows(), 1);
    assert_eq!(repeated_lookup.nrows(), 2);
    assert_eq!(first_lookup.row(0), repeated_lookup.row(0));
    assert_eq!(repeated_lookup.row(0), repeated_lookup.row(1));

    let token_vectors = DMatrix::from_row_slice(2, 2, &[1.0, 2.0, 3.0, 4.0]);
    let position_vectors = DMatrix::from_row_slice(2, 2, &[0.5, 0.5, -1.0, 1.0]);
    let combined =
        add_absolute_positions(&token_vectors, &position_vectors).expect("matching shapes");
    assert_eq!(
        combined,
        DMatrix::from_row_slice(2, 2, &[1.5, 2.5, 2.0, 5.0])
    );
}

#[test]
fn candle_and_nalgebra_paths_produce_the_same_input_embeddings() {
    let token_table = EmbeddingTable::seeded(8, 3, 123);
    let position_table = EmbeddingTable::seeded(4, 3, 999);
    let token_ids = [2, 3, 5, 1];
    let position_ids = [0, 1, 2, 3];

    let expected = add_absolute_positions(
        &token_table.lookup(&token_ids).expect("valid token IDs"),
        &position_table
            .lookup(&position_ids)
            .expect("valid position IDs"),
    )
    .expect("matching matrix shapes");
    let candle = candle_input_embeddings(&token_table, &position_table, &token_ids)
        .expect("Candle input construction succeeds");
    assert_eq!(candle.dims(), &[4, 3]);
    let candle_values = candle.to_vec2::<f32>().expect("Candle values materialize");
    let expected_values = expected
        .row_iter()
        .map(|row| row.iter().copied().collect::<Vec<_>>())
        .collect::<Vec<_>>();
    assert_eq!(candle_values, expected_values);
}

fn gpt2_tokenizer() -> Gpt2BpeTokenizer {
    let asset_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("assets/gpt2");
    Gpt2BpeTokenizer::from_files(asset_dir.join("encoder.json"), asset_dir.join("vocab.bpe"))
        .expect("released GPT-2 tokenizer artifacts load")
}

#[test]
fn gpt2_byte_mapping_is_bijective_for_all_bytes() {
    let (byte_to_unicode, unicode_to_byte) = gpt2_byte_to_unicode();
    assert_eq!(byte_to_unicode.len(), 256);
    assert_eq!(unicode_to_byte.len(), 256);
    (0_u8..=255).for_each(|byte| {
        let character = byte_to_unicode[&byte];
        assert_eq!(unicode_to_byte[&character], byte);
    });
}

#[test]
fn gpt2_bpe_matches_reference_token_ids_and_preserves_special_tokens() {
    let tokenizer = gpt2_tokenizer();
    assert_eq!(tokenizer.vocabulary_size(), 50_257);
    assert_eq!(
        tokenizer
            .encode("Hello world!")
            .expect("reference text encodes"),
        vec![15496, 995, 0]
    );
    assert_eq!(
        tokenizer
            .encode("Akwirw ier")
            .expect("reference text encodes"),
        vec![33901, 86, 343, 86, 220, 959]
    );
    assert_eq!(tokenizer.special_token_id(END_OF_TEXT), Some(50_256));
    assert_eq!(
        tokenizer
            .encode("one<|endoftext|>two")
            .expect("special token encodes"),
        vec![505, 50_256, 11545]
    );
}

#[test]
fn gpt2_bpe_round_trips_unicode_and_repeated_special_tokens() {
    let tokenizer = gpt2_tokenizer();
    let text = "Café 🤖<|endoftext|>次の文<|endoftext|>done";
    let ids = tokenizer.encode(text).expect("UTF-8 text encodes");
    assert_eq!(tokenizer.decode(&ids).expect("IDs decode"), text);
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
