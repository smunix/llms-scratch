# LLMs from Scratch — Rust Study Materials

This private study repository contains original Rust implementations and explanations written while reading *Build a Large Language Model (From Scratch)* by Sebastian Raschka. The materials focus on understanding concepts through small, runnable programs; they do not redistribute the book or its datasets.

## Contents

| Folder | Coverage | Getting started |
|---|---|---|
| [`chapter-02`](chapter-02) | Text data preparation: tokenization, token IDs, special tokens, toy and GPT-2-compatible byte-level BPE, sliding-window supervision, and a Candle-only embedding/position pipeline. | Read `chapter-02/README.md`, run `cargo test`, then use `cargo run --bin embeddings_and_positions`. |
| [`chapter-03`](chapter-03) | Attention mechanisms: Candle-based scaled dot-product attention, causal masking, efficient head splitting, multi-head recombination, and output projection. | Read `chapter-03/README.md`, run `cargo test --all-targets`, then use `cargo run --bin multi_head_attention`. |
| [`chapter-04`](chapter-04) | GPT transformer layers: Candle layer normalization, causal multi-head attention, GELU feed-forward layers, residual connections, and pre-layer-normalized transformer blocks. | Read `chapter-04/README.md`, run `cargo test --all-targets`, then use `cargo run --bin transformer_block`. |

## Conventions

Each chapter folder contains a technical study guide, runnable Rust examples, and tests. The code emphasizes transparent data transformations over production performance. When a future chapter depends on a production model component, its documentation should state the simplification and identify the compatibility boundary.

## Reference

Sebastian Raschka, *Build a Large Language Model (From Scratch)*, Manning, 2025. [Official book page][1]

[1]: https://www.manning.com/books/build-a-large-language-model-from-scratch
