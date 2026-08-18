use chapter_02_text_data::apply_bpe_merges;
use itertools::Itertools;

fn main() {
    // A toy starting alphabet and a hypothetical merge list learned from corpus frequencies.
    // This illustrates the merge operation only; GPT-style BPE also operates on bytes and
    // needs a much larger learned vocabulary plus special-token handling.
    let symbols = "l o w e r"
        .split_whitespace()
        .map(str::to_owned)
        .collect_vec();
    let merges = [("l", "o"), ("lo", "w"), ("e", "r")];

    let result = apply_bpe_merges(symbols.clone(), &merges);
    println!("Initial symbols: {}", symbols.iter().join(" | "));
    println!("Applied merges: {merges:?}");
    println!("Final subwords: {}", result.iter().join(" | "));

    let unfamiliar = "Akwirw"
        .chars()
        .map(|character| character.to_string())
        .collect_vec();
    let fallback = apply_bpe_merges(unfamiliar.clone(), &[]);
    println!(
        "\nUnknown spelling without matching merges: {}",
        fallback.iter().join(" | ")
    );
    println!(
        "Character-level fallback is why BPE can represent text beyond a word-only vocabulary."
    );
}
