# CLAUDE.md

## Project Overview

This repository implements a **Retrieval-Augmented Generation (RAG)** tool built with LangChain. It loads PDF documents, chunks them, stores embeddings in a Milvus vector database, and answers questions using a Mistral-7B LLM via Hugging Face.

The project can be used as a **CLI tool** or explored interactively via the original **Jupyter notebook**.

**License:** MIT (Copyright 2025 BatHands)

## Repository Structure

```
AI/
├── main.py                        # CLI entrypoint
├── src/
│   ├── __init__.py
│   ├── config.py                  # Configuration (env vars, model IDs, defaults)
│   ├── loader.py                  # DoclingPDFLoader + load_and_split()
│   └── chain.py                   # Vector store + RAG chain assembly
├── RAG_with_LangChain.ipynb       # Original Colab notebook (demo/reference)
├── requirements.txt               # Python dependencies
├── .env.example                   # Environment variable template
├── .gitignore                     # Git ignore rules
├── CLAUDE.md                      # This file — AI assistant guide
└── LICENSE                        # MIT License
```

## Technology Stack

| Component        | Technology                                            |
|------------------|-------------------------------------------------------|
| Language         | Python 3.10+                                          |
| RAG Framework    | LangChain (LCEL chain composition)                    |
| Document Loading | Docling (`DocumentConverter` -> markdown)             |
| Text Splitting   | `RecursiveCharacterTextSplitter`                      |
| Embeddings       | Hugging Face `BAAI/bge-small-en-v1.5`                |
| Vector Store     | Milvus Lite (local file-based, via `langchain-milvus`)|
| LLM              | `mistralai/Mistral-7B-Instruct-v0.3` via HF API      |
| Env Management   | `python-dotenv`                                       |

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Configure environment
cp .env.example .env
# Edit .env and add your HF_API_KEY

# 3. Query a PDF
python main.py --pdf document.pdf --query "Summarize this document"

# Also works with URLs
python main.py --pdf https://example.com/paper.pdf --query "What are the key findings?"
```

## Environment Variables

| Variable           | Required | Description                                | Default                              |
|--------------------|----------|--------------------------------------------|--------------------------------------|
| `HF_API_KEY`       | Yes      | Hugging Face API token for model inference | None                                 |
| `HF_EMBED_MODEL_ID`| No      | Embedding model ID                         | `BAAI/bge-small-en-v1.5`            |
| `HF_LLM_MODEL_ID` | No       | LLM model ID                              | `mistralai/Mistral-7B-Instruct-v0.3`|
| `MILVUS_URI`       | No       | Milvus connection URI                      | `./milvus_store/milvus.db`           |
| `CHUNK_SIZE`       | No       | Text chunk size in characters              | `1000`                               |
| `CHUNK_OVERLAP`    | No       | Overlap between chunks in characters       | `200`                                |

Place these in a `.env` file at the project root. See `.env.example` for a template.

## Code Architecture

### `src/config.py`
Central configuration module. Loads `.env` and exposes all settings as module-level constants. All other modules import from here.

### `src/loader.py`
- **`DoclingPDFLoader`** — Custom LangChain `BaseLoader` that converts PDFs to markdown via Docling's `DocumentConverter`, then yields `LCDocument` objects.
- **`load_and_split()`** — Convenience function that loads PDF(s) and splits into chunks using `RecursiveCharacterTextSplitter`.

### `src/chain.py`
- **`build_vectorstore()`** — Creates a Milvus vector store from document splits using HuggingFace embeddings.
- **`build_rag_chain()`** — Assembles the full RAG chain: retriever -> prompt -> ChatHuggingFace LLM -> string output parser.

### `main.py`
CLI entrypoint using `argparse`. Validates inputs, then runs the full pipeline: load -> split -> embed -> query.

### `RAG_with_LangChain.ipynb`
Original Colab notebook. Kept as a demo/reference. Contains the same pipeline in an interactive cell-by-cell format.

## Key Code Conventions

- **LCEL chain composition:** The RAG chain uses LangChain Expression Language with the pipe (`|`) operator.
- **Prompt structure:** `ChatPromptTemplate` with separate system and human messages. The human template injects `{context}` and `{question}`.
- **Type hints:** Python type hints throughout (e.g., `str | list[str]`, `Iterator[LCDocument]`).
- **Lazy imports in CLI:** Heavy dependencies are imported inside `main()` so argument validation and error messages are fast.

## Guidelines for AI Assistants

- **Do not commit secrets.** Never add `.env` files or API keys to the repository.
- **Update requirements.txt** when adding new dependencies.
- **Keep config centralized** in `src/config.py`. New settings should follow the `os.environ.get()` pattern with sensible defaults.
- **Maintain the notebook.** The Colab notebook should remain functional as a standalone demo, even though the CLI is the primary interface.
- **Type hints required.** All new functions should include type hints.
- **Environment variables** for any new secrets or configuration. Never hardcode credentials or API keys.
