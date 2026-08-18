# LLMs from Scratch — Rust Study Materials

This private study repository contains original Rust implementations and explanations written while reading *Build a Large Language Model (From Scratch)* by Sebastian Raschka. The materials focus on understanding concepts through small, runnable programs; they do not redistribute the book or its datasets.

## Contents

| Folder | Coverage | Getting started |
|---|---|---|
| [`chapter-02`](chapter-02) | Text data preparation: tokenization, token IDs, special tokens, toy BPE merge application, sliding-window supervision, token embeddings, and absolute positional embeddings. | Read `chapter-02/README.md`, then run `cargo test` from that directory. |

## Conventions

Each chapter folder contains a technical study guide, runnable Rust examples, and tests. The code emphasizes transparent data transformations over production performance. When a future chapter depends on a production model component, its documentation should state the simplification and identify the compatibility boundary.

## Reference

Sebastian Raschka, *Build a Large Language Model (From Scratch)*, Manning, 2025. [Official book page][1]

[1]: https://www.manning.com/books/build-a-large-language-model-from-scratch
