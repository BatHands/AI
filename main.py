#!/usr/bin/env python3
"""CLI tool for querying PDF documents using RAG."""

import argparse
import sys
import os


def main():
    parser = argparse.ArgumentParser(
        description="Ask questions about PDF documents using RAG with LangChain."
    )
    parser.add_argument(
        "--pdf",
        required=True,
        help="Path or URL to a PDF file to query.",
    )
    parser.add_argument(
        "--query",
        required=True,
        help="Question to ask about the document.",
    )
    args = parser.parse_args()

    # Validate local file paths exist
    if not args.pdf.startswith(("http://", "https://")) and not os.path.isfile(args.pdf):
        print(f"Error: file not found: {args.pdf}", file=sys.stderr)
        sys.exit(1)

    from src.config import HF_API_KEY
    if not HF_API_KEY:
        print(
            "Error: HF_API_KEY is not set. "
            "Add it to a .env file or export it as an environment variable.",
            file=sys.stderr,
        )
        sys.exit(1)

    from src.loader import load_and_split
    from src.chain import build_vectorstore, build_rag_chain

    print(f"Loading and processing: {args.pdf}")
    splits = load_and_split(args.pdf)
    print(f"Created {len(splits)} chunks.")

    print("Building vector store...")
    vectorstore = build_vectorstore(splits)

    print("Assembling RAG chain...")
    rag_chain = build_rag_chain(vectorstore)

    print(f"\nQuery: {args.query}\n")
    answer = rag_chain.invoke(args.query)
    print(f"Answer: {answer}")


if __name__ == "__main__":
    main()
