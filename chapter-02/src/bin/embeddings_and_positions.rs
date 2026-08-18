use chapter_02_text_data::{add_absolute_positions, candle_input_embeddings, EmbeddingTable};
use itertools::Itertools;

fn main() -> Result<(), String> {
    // Rows are token IDs, columns are embedding dimensions. Training would optimize the values.
    let token_table = EmbeddingTable::seeded(8, 3, 123);
    let position_table = EmbeddingTable::seeded(4, 3, 999);
    let input_ids = vec![2, 3, 5, 1];
    let position_ids = (0..input_ids.len()).collect_vec();

    // The explicit Candle path: indexed lookup followed by elementwise positional addition.
    let token_embeddings = token_table.lookup(&input_ids)?;
    let position_embeddings = position_table.lookup(&position_ids)?;
    let explicit_input = add_absolute_positions(&token_embeddings, &position_embeddings)?;

    // The compact Candle path: the same sequence construction in one helper call.
    let helper_input = candle_input_embeddings(&token_table, &position_table, &input_ids)?;
    let explicit_values = explicit_input
        .to_vec2::<f32>()
        .map_err(|error| format!("could not materialize explicit Candle tensor: {error}"))?;
    let helper_values = helper_input
        .to_vec2::<f32>()
        .map_err(|error| format!("could not materialize helper Candle tensor: {error}"))?;

    println!("Input token IDs: {input_ids:?}");
    println!(
        "Candle token embedding tensor shape: {:?}",
        token_embeddings.dims()
    );
    println!(
        "Candle position embedding tensor shape: {:?}",
        position_embeddings.dims()
    );
    println!("Candle input embedding shape: {:?}", helper_input.dims());
    println!("Candle input embedding values: {helper_values:?}");

    (explicit_values == helper_values)
        .then_some(())
        .ok_or_else(|| "explicit and helper Candle paths disagree".to_owned())?;
    println!("The explicit and helper Candle paths agree exactly.");
    Ok(())
}
