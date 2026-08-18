use chapter_02_text_data::{Gpt2BpeTokenizer, END_OF_TEXT};
use itertools::Itertools;
use std::path::PathBuf;

fn main() -> Result<(), String> {
    let asset_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("assets/gpt2");
    let tokenizer =
        Gpt2BpeTokenizer::from_files(asset_dir.join("encoder.json"), asset_dir.join("vocab.bpe"))?;

    println!("GPT-2 vocabulary size: {}", tokenizer.vocabulary_size());
    println!(
        "{END_OF_TEXT} ID: {:?}",
        tokenizer.special_token_id(END_OF_TEXT)
    );

    let samples = [
        "Hello world!",
        "Akwirw ier",
        "Café 🤖 — byte-level BPE round-trips UTF-8.",
        "First document.<|endoftext|>Second document.",
    ];

    samples.iter().try_for_each(|text| {
        let ids = tokenizer.encode(text)?;
        let pieces = ids
            .iter()
            .map(|id| {
                tokenizer
                    .token_for_id(*id)
                    .unwrap_or("<missing>")
                    .escape_default()
                    .to_string()
            })
            .join(" | ");
        let decoded = tokenizer.decode(&ids)?;

        println!("\nInput:    {text:?}");
        println!("IDs:      {ids:?}");
        println!("BPE text: {pieces}");
        println!("Decoded:  {decoded:?}");
        (decoded == *text)
            .then_some(())
            .ok_or_else(|| "round-trip check failed".to_owned())
    })
}
