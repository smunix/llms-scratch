# GPT-2 tokenizer fixtures

This directory contains the released tokenizer artifacts used by `Gpt2BpeTokenizer` and its integration tests. They are included to make the Chapter 2 byte-level BPE implementation self-contained and to validate that the emitted IDs match the GPT-2 tokenizer contract. They are **not** model weights.

| File | Role | SHA-256 |
|---|---|---|
| `encoder.json` | Maps serialized byte-level BPE tokens to GPT-2 token IDs. | `196139668be63f3b5d6574427317ae82f612a97c5d1cdaf36ed2256dbf636783` |
| `vocab.bpe` | Ordered BPE merge pairs; earlier lines have lower merge rank. | `1ce1664773c50f3e0cc8842619a93edc4624525b728b188a9e0be33b7726adc5` |

The files were retrieved from the released 124M GPT-2 model artifact location on 2026-08-18. The GPT-2 model card documents a byte-level BPE tokenizer with a vocabulary size of 50,257 and an MIT license. [1]

> **Compatibility requirement.** These files must be paired with the matching GPT-2 embedding table. Replacing either artifact changes the generated token IDs and breaks compatibility with a pretrained checkpoint.

## Source

[1] GPT-2 model card, “openai-community/gpt2,” Hugging Face. [Model card][1]

[1]: https://huggingface.co/openai-community/gpt2
