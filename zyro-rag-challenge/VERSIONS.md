# Zyro Dynamics HR RAG Challenge — Version History

## Version 2 (Current) — `Starter_Notebook_v2.ipynb`
**Date**: 2026-06-10
**Score**: TBD (targeting 90+)

### Changes from V1
| Component | V1 | V2 | Reason |
|---|---|---|---|
| Embedding Model | `all-MiniLM-L6-v2` | `BAAI/bge-large-en-v1.5` | 3x better retrieval accuracy on benchmarks |
| Chunk Size | 1000 chars / 200 overlap | 750 chars / 150 overlap | More focused embeddings, less noise |
| Retriever k | 3 | 5 | Captures answers spread across multiple docs |
| RAG Prompt | Basic instructions | Strict anti-hallucination rules | Prevents LLM from using outside knowledge |
| Guardrail REFUSAL_MESSAGE | Multi-line with `\` continuation | Single-line string | Fixed SyntaxError on Kaggle paste |
| `ask_bot` function | Missing from Cell 11 | Properly defined | Fixed NameError in Cell 12 |

### Bug Fixes
- Fixed `REFUSAL_MESSAGE` SyntaxError caused by hidden spaces after line continuation characters
- Fixed `NameError: name 'ask_bot' is not defined` — function was named `check_guardrail` but Cell 12 expected `ask_bot`
- Fixed `CORPUS_PATH` — Kaggle competition path differs from direct upload path

---

## Version 1 — `Starter_Notebook_v1.ipynb`
**Date**: 2026-06-10
**Score**: 81.10 (Rank #2)

### Configuration
- Embedding: `sentence-transformers/all-MiniLM-L6-v2`
- Chunks: 1000 chars, 200 overlap
- Retriever: k=3
- LLM: `llama-3.3-70b-versatile` via Groq
- Temperature: 0.1

### Known Issues
- REFUSAL_MESSAGE had line continuation syntax error when pasted into Kaggle
- Cell 11 defined `check_guardrail` but Cell 12 called `ask_bot` (NameError)
- Required manual CORPUS_PATH correction for competition dataset
