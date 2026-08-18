use chapter_02_text_data::{build_vocab, simple_tokenize, SimpleTokenizer, END_OF_TEXT, UNK};

fn main() -> Result<(), String> {
    let training_text = "Hello, world. This is a tiny training corpus.";
    let tokens = simple_tokenize(training_text);
    println!("Tokens: {tokens:?}");

    let strict_vocab = build_vocab(&tokens, &[]);
    let strict = SimpleTokenizer::strict(strict_vocab);
    let ids = strict.encode("Hello, world.")?;
    println!("Strict encode: {ids:?}");
    println!("Strict decode: {}", strict.decode(&ids)?);

    match strict.encode("Hello, Rustacean!") {
        Ok(unexpected) => println!("Unexpected strict result: {unexpected:?}"),
        Err(error) => println!("Strict tokenizer correctly rejects an unseen token: {error}"),
    }

    let vocabulary = build_vocab(&tokens, &[END_OF_TEXT, UNK]);
    let tokenizer = SimpleTokenizer::with_unknown(vocabulary, UNK)?;
    let combined = format!("Hello, Rustacean! {END_OF_TEXT} This is a corpus.");
    let robust_ids = tokenizer.encode(&combined)?;

    println!("Vocabulary size including special tokens: {}", tokenizer.vocab_size());
    println!("Fallback encode: {robust_ids:?}");
    println!("Fallback decode: {}", tokenizer.decode(&robust_ids)?);
    Ok(())
}
