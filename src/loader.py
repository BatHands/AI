from typing import Iterator

from langchain_core.document_loaders import BaseLoader
from langchain_core.documents import Document as LCDocument
from langchain_text_splitters import RecursiveCharacterTextSplitter

from docling.document_converter import DocumentConverter

from src.config import CHUNK_SIZE, CHUNK_OVERLAP


class DoclingPDFLoader(BaseLoader):
    """Loads PDF files and converts them to markdown using Docling."""

    def __init__(self, file_path: str | list[str]) -> None:
        self._file_paths = file_path if isinstance(file_path, list) else [file_path]
        self._converter = DocumentConverter()

    def lazy_load(self) -> Iterator[LCDocument]:
        for source in self._file_paths:
            dl_doc = self._converter.convert(source).document
            text = dl_doc.export_to_markdown()
            yield LCDocument(page_content=text)


def load_and_split(file_path: str | list[str]) -> list[LCDocument]:
    """Load PDF(s) and split into chunks."""
    loader = DoclingPDFLoader(file_path=file_path)
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )
    docs = loader.load()
    return text_splitter.split_documents(docs)
