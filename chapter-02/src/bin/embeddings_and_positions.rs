use chapter_02_text_data::{add_absolute_positions, EmbeddingTable};

fn main() -> Result<(), String> {
    // Rows are token IDs, columns are embedding dimensions. Training would optimize the values.
    let token_table = EmbeddingTable::seeded(8, 3, 123);
    let position_table = EmbeddingTable::seeded(4, 3, 999);
    let input_ids = vec![2, 3, 5, 1];
    let position_ids = vec![0, 1, 2, 3];

    let token_embeddings = token_table.lookup(&input_ids)?;
    let position_embeddings = position_table.lookup(&position_ids)?;
    let input_embeddings = add_absolute_positions(&token_embeddings, &position_embeddings)?;

    println!("Input token IDs: {input_ids:?}");
    println!("Token embedding matrix (sequence × dimensions): {token_embeddings:?}");
    println!("Position embedding matrix: {position_embeddings:?}");
    println!("Final input embedding matrix = token + position: {input_embeddings:?}");
    println!(
        "Shape: {} positions × {} dimensions",
        input_embeddings.len(),
        input_embeddings.first().map_or(0, Vec::len)
    );
    Ok(())
}
