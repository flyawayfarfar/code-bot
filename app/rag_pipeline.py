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
def make_embedder():
    provider = settings.embedding_provider
    if provider == "google":
        from langchain_google_genai import GoogleGenerativeAIEmbeddings
        if not settings.google_api_key:
            raise RuntimeError("GOOGLE_API_KEY not set")
        return GoogleGenerativeAIEmbeddings(
            model="models/text-embedding-004", 
            google_api_key=settings.google_api_key
        )
    
    if provider == "openai":
        from langchain_community.embeddings import OpenAIEmbeddings
        if not settings.openai_api_key:
            raise RuntimeError("OPENAI_API_KEY not set")
        return OpenAIEmbeddings(openai_api_key=settings.openai_api_key)

    # Fallback to Ollama
    from langchain_community.embeddings import OllamaEmbeddings
    return OllamaEmbeddings(
        model=settings.ollama_embed_model,
        base_url=settings.ollama_base_url,
    )


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
    
    vectordb = Chroma(
        persist_directory=settings.active_chroma_dir,
        embedding_function=embeddings,
        collection_name="local-rag",
    )

    # Batch processing to avoid Google API rate limits (especially free tier)
    # Rate limit is often ~60 requests/min or lower for embeddings
    # Using extremely conservative settings: 1 chunk at a time, 2s sleep
    batch_size = 1
    import time

    print(f"Adding documents in batches of {batch_size}...")
    for i in range(0, len(chunks), batch_size):
        batch = chunks[i : i + batch_size]
        try:
            vectordb.add_documents(batch)
            print(f"Processed {min(i + batch_size, len(chunks))}/{len(chunks)}")
            # Sleep to be safe for free tier
            if settings.embedding_provider == "google":
                time.sleep(2.0)
        except Exception as e:
            print(f"Error processing batch {i}: {e}")
            # simple retry with backoff
            print("Retrying after 30s...")
            time.sleep(30)
            vectordb.add_documents(batch)

    vectordb.persist()
    return vectordb
