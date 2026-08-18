use chapter_02_text_data::{add_absolute_positions, candle_input_embeddings, EmbeddingTable};
use itertools::Itertools;

fn main() -> Result<(), String> {
    // Rows are token IDs, columns are embedding dimensions. Training would optimize the values.
    let token_table = EmbeddingTable::seeded(8, 3, 123);
    let position_table = EmbeddingTable::seeded(4, 3, 999);
    let input_ids = vec![2, 3, 5, 1];
    let position_ids = (0..input_ids.len()).collect_vec();

    // The transparent linear-algebra view: explicit nalgebra matrices and elementwise addition.
    let token_embeddings = token_table.lookup(&input_ids)?;
    let position_embeddings = position_table.lookup(&position_ids)?;
    let nalgebra_input_embeddings =
        add_absolute_positions(&token_embeddings, &position_embeddings)?;

    // The tensor-programming view: Candle performs indexed lookup and positional addition on CPU.
    let candle_input = candle_input_embeddings(&token_table, &position_table, &input_ids)?;
    let candle_values = candle_input
        .to_vec2::<f32>()
        .map_err(|error| format!("could not materialize Candle tensor: {error}"))?;
    let nalgebra_values = nalgebra_input_embeddings
        .row_iter()
        .map(|row| row.iter().copied().collect_vec())
        .collect_vec();

    println!("Input token IDs: {input_ids:?}");
    println!("nalgebra token embedding matrix:\n{token_embeddings}");
    println!("nalgebra position embedding matrix:\n{position_embeddings}");
    println!("nalgebra input embeddings = token + position:\n{nalgebra_input_embeddings}");
    println!(
        "Candle input-embedding tensor shape: {:?}",
        candle_input.dims()
    );
    println!("Candle input-embedding values: {candle_values:?}");

    (nalgebra_values == candle_values)
        .then_some(())
        .ok_or_else(|| "nalgebra and Candle paths disagree".to_owned())?;
    println!("The nalgebra and Candle paths agree exactly.");
    Ok(())
}
