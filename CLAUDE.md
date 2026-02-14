# CLAUDE.md

## Project Overview

This repository contains a **Retrieval-Augmented Generation (RAG)** proof-of-concept built with LangChain. The single Jupyter notebook demonstrates a pipeline that loads a PDF document, chunks it, stores embeddings in a Milvus vector database, and answers questions using a Mistral-7B LLM via Hugging Face.

**License:** MIT (Copyright 2025 BatHands)

## Repository Structure

```
AI/
├── CLAUDE.md                      # This file — AI assistant guide
├── LICENSE                        # MIT License
└── RAG_with_LangChain.ipynb       # Main notebook (RAG pipeline)
```

This is a minimal, single-notebook project. There is no Python package structure, no tests, no CI/CD, and no build system.

## Technology Stack

| Component         | Technology                              |
|--------------------|-----------------------------------------|
| Language           | Python (Jupyter Notebook)               |
| Runtime            | Google Colab (primary), local Jupyter   |
| RAG Framework      | LangChain                               |
| Document Loading   | Docling (`DocumentConverter`)           |
| Text Splitting     | `RecursiveCharacterTextSplitter` (chunk_size=1000, overlap=200) |
| Embeddings         | Hugging Face `BAAI/bge-small-en-v1.5`  |
| Vector Store       | Milvus (via `langchain-milvus`, lite mode by default) |
| LLM                | `mistralai/Mistral-7B-Instruct-v0.3` via Hugging Face Inference API |
| Env Management     | `python-dotenv`                         |

## Notebook Pipeline (RAG_with_LangChain.ipynb)

The notebook executes a linear RAG pipeline across 12 cells:

1. **Dependencies** — Installs packages via `%pip install`
2. **Environment** — Loads `.env` file with `dotenv`
3. **DoclingPDFLoader** — Custom `BaseLoader` subclass that converts PDFs to markdown using Docling
4. **Source document** — Points to a sample PDF URL from Trinity College
5. **Text splitter** — Configures `RecursiveCharacterTextSplitter`
6. **Load & split** — Loads the PDF and splits into chunks
7. **HF login** — Interactive Hugging Face authentication via `notebook_login()`
8. **Embeddings** — Initializes BGE-small embedding model
9. **Vector store** — Creates a Milvus collection from document splits
10. **LLM setup** — Configures Mistral-7B via `HuggingFaceEndpoint`
11. **RAG chain** — Assembles retriever, prompt template, LLM, and output parser into a LangChain chain
12. **Query** — Runs a sample question against the RAG chain

## Environment Variables

| Variable     | Required | Description                                      | Default                          |
|--------------|----------|--------------------------------------------------|----------------------------------|
| `HF_API_KEY` | Yes      | Hugging Face API token for model inference        | None                             |
| `MILVUS_URI` | No       | Milvus connection URI                             | Temporary local SQLite-like file |

These should be placed in a `.env` file at the project root, or set as environment variables in Colab.

## Development Workflow

### Running the notebook

- **Google Colab (recommended):** Open via the Colab badge in the notebook's first cell. Dependencies install inline.
- **Local Jupyter:** Ensure Python 3.10+ is available. Create a `.env` file with `HF_API_KEY`, then run cells sequentially.

### Dependencies

All dependencies are installed inline in the notebook via pip. There is no `requirements.txt` or `pyproject.toml`. The core packages are:

```
docling python-dotenv langchain langchain-milvus langchain-text-splitters langchain-huggingface pymilvus[milvus_lite] huggingface_hub ipywidgets
```

### No build, test, or lint commands

This project has no:
- Build system or compilation step
- Test suite or test runner
- Linter or formatter configuration
- CI/CD pipeline
- Makefile or scripts

## Key Code Conventions

- **Custom loader pattern:** `DoclingPDFLoader` extends LangChain's `BaseLoader` and implements `lazy_load()` to yield `LCDocument` objects with markdown content.
- **Chain composition:** The RAG chain uses LangChain Expression Language (LCEL) with the pipe (`|`) operator to compose retriever, prompt, model, and parser.
- **Prompt structure:** Uses `ChatPromptTemplate` with separate system and human messages. The human message template injects `{context}` and `{question}` placeholders.
- **Type hints:** The code uses Python type hints (e.g., `str | list[str]`, `Iterator[LCDocument]`).

## Guidelines for AI Assistants

- **Do not commit secrets.** Never add `.env` files or API keys to the repository.
- **Preserve notebook structure.** The cells follow a logical pipeline order. New cells should maintain this linear flow.
- **Keep it simple.** This is a proof-of-concept. Avoid over-engineering with unnecessary abstractions, classes, or config files.
- **Colab compatibility.** Any changes must remain runnable in Google Colab. Avoid local-only dependencies or file paths.
- **Inline installs.** Dependencies are installed in notebook cells with `%pip install`. If adding a new dependency, add it to the existing install cell rather than creating a new one.
- **Environment variables.** Any new secrets or configuration should use `os.environ.get()` with sensible defaults where possible, loaded via `dotenv`.
