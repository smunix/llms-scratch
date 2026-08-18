use chapter_02_text_data::sliding_window_examples;
use itertools::Itertools;

fn main() -> Result<(), String> {
    // Think of each number as a token ID already produced by a tokenizer.
    let token_ids = vec![40, 367, 2885, 1464, 1807, 3619, 402, 271, 10];
    let context_length = 4;

    let overlapping = sliding_window_examples(&token_ids, context_length, 1)?;
    println!("stride = 1: overlapping windows");
    overlapping
        .iter()
        .take(3)
        .enumerate()
        .for_each(|(index, example)| {
            println!(
                "  example {index}: input {:?} -> targets {:?}",
                example.input_ids, example.target_ids
            );
        });

    let adjacent = sliding_window_examples(&token_ids, context_length, context_length)?;
    println!("\nstride = context length: adjacent windows");
    adjacent.iter().enumerate().for_each(|(index, example)| {
        println!(
            "  example {index}: input [{}] -> targets [{}]",
            example.input_ids.iter().join(", "),
            example.target_ids.iter().join(", ")
        );
    });

    println!("\nEach target row is its input row shifted one token to the left.");
    Ok(())
}
