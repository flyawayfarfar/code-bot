# app/rag_pipeline.py
import os
from typing import List

from app.config import settings

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma

from langchain_community.document_loaders import (
    DirectoryLoader,
    PyPDFLoader,
    TextLoader,
)


class SafeTextLoader(TextLoader):
    """Text loader that tolerates encoding issues."""

    def __init__(self, file_path: str, encoding: str | None = "utf-8") -> None:
        super().__init__(file_path, encoding=encoding, autodetect_encoding=True)

    def load(self) -> List[Document]:
        try:
            return super().load()
        except Exception as exc:  # pragma: no cover - diagnostic path
            encoding = self.encoding or "utf-8"
            print(f"SafeTextLoader: falling back to permissive read for {self.file_path} ({exc})")
            with open(self.file_path, "r", encoding=encoding, errors="ignore") as handle:
                text = handle.read()
            return [Document(page_content=text, metadata={"source": self.file_path})]


# Embeddings
if settings.use_local_embeddings:
    from langchain_community.embeddings import OllamaEmbeddings

    def make_embedder():
        return OllamaEmbeddings(
            model=settings.ollama_embed_model,
            base_url=settings.ollama_base_url,
        )
else:
    from langchain_community.embeddings import OpenAIEmbeddings

    def make_embedder():
        if not settings.openai_api_key:
            raise RuntimeError("OPENAI_API_KEY not set")
        return OpenAIEmbeddings(openai_api_key=settings.openai_api_key)


def load_documents(data_dir: str) -> List[Document]:
    docs: List[Document] = []

    # PDFs
    try:
        pdf_loader = DirectoryLoader(
            data_dir,
            glob="**/*.pdf",
            loader_cls=PyPDFLoader,
            recursive=True,
        )
        docs.extend(pdf_loader.load())
    except Exception as exc:  # pragma: no cover - diagnostic path
        print("PDF load error:", exc)

    # Plain text / markdown / csv as text
    for pattern in ("**/*.txt", "**/*.md", "**/*.csv"):
        try:
            loader = DirectoryLoader(
                data_dir,
                glob=pattern,
                loader_cls=SafeTextLoader,
                recursive=True,
            )
            docs.extend(loader.load())
        except Exception as exc:  # pragma: no cover - diagnostic path
            print(f"Load error for {pattern}:", exc)

    # If no loaders matched, try the simplest fallback (files in root)
    if not docs and os.path.isdir(data_dir):
        for fname in os.listdir(data_dir):
            fpath = os.path.join(data_dir, fname)
            if os.path.isfile(fpath):
                try:
                    with open(fpath, "r", encoding="utf-8", errors="ignore") as handle:
                        docs.append(
                            Document(
                                page_content=handle.read(),
                                metadata={"source": fpath},
                            )
                        )
                except Exception as exc:  # pragma: no cover - diagnostic path
                    print("Fallback read error:", fpath, exc)

    return docs


def chunk_documents(docs: List[Document]) -> List[Document]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=900,
        chunk_overlap=150,
        separators=["\n\n", "\n", " ", ""],
    )
    return splitter.split_documents(docs)


def build_vector_store() -> Chroma:
    docs = load_documents(settings.data_dir)
    if not docs:
        print("No documents found in", settings.data_dir)
        raise SystemExit(0)

    chunks = chunk_documents(docs)
    print(f"Loaded {len(docs)} docs -> {len(chunks)} chunks")

    embeddings = make_embedder()
    vectordb = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=settings.active_chroma_dir,
        collection_name="local-rag",
    )
    vectordb.persist()
    return vectordb
