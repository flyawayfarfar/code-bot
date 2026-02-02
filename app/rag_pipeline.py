# app/rag_pipeline.py
import os
import re
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

import logging
# Configure logger for indexing
indexing_logger = logging.getLogger("indexing")
indexing_logger.setLevel(logging.INFO)
# Clear existing handlers
if indexing_logger.handlers:
    indexing_logger.handlers.clear()
file_handler = logging.FileHandler("indexing.log", mode="w", encoding="utf-8")
formatter = logging.Formatter('%(asctime)s [%(levelname)s] %(message)s')
file_handler.setFormatter(formatter)
indexing_logger.addHandler(file_handler)
# Also add console handler
console_handler = logging.StreamHandler()
console_handler.setFormatter(formatter)
indexing_logger.addHandler(console_handler)


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
            model="models/gemini-embedding-001", 
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



def preprocess_code(content: str, ext: str) -> str:
    """Strips comments and noise based on file extension."""
    if ext in {".java", ".xml", ".xsd", ".config", ".yml", ".yaml"}:
        # Strip block comments /* ... */ or <!-- ... -->
        content = re.sub(r'/\*.*?\*/', '', content, flags=re.DOTALL)
        content = re.sub(r'<!--.*?-->', '', content, flags=re.DOTALL)
        # Strip single line // (but NOT if it's like https://)
        if ext == ".java":
             # Match // only if not preceded by :
             content = re.sub(r'(?<!:)//.*', '', content)
        if ext in {".yml", ".yaml", ".properties"}:
             content = re.sub(r'#.*', '', content)
    elif ext == ".py":
        # Strip triple quotes
        content = re.sub(r'"{3}.*?"{3}', '', content, flags=re.DOTALL)
        content = re.sub(r"'{3}.*?'{3}", '', content, flags=re.DOTALL)
        # Strip # comments
        content = re.sub(r'#.*', '', content)
    
    # Strip excessive whitespace
    content = "\n".join(line for line in content.splitlines() if line.strip())
    return content


def load_documents(data_dir: str) -> List[Document]:
    """
    Implements recursive project discovery. 
    Immediate subdirectories of data_dir are treated as "Projects".
    """
    all_docs: List[Document] = []
    
    # Extensions to track
    EXTENSIONS = {
        ".java", ".xml", ".yml", ".yaml", ".properties", 
        ".xsd", ".md", ".txt", ".config", "pom.xml"
    }
    
    # Folders to ignore
    FORBIDDEN_FOLDERS = {"target", ".git", ".idea", ".vscode", "bin", "build", "node_modules"}

    if not os.path.exists(data_dir):
        print(f"Data directory {data_dir} does not exist.")
        return []

    # Get immediate subdirectories as projects
    projects = [d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))]
    
    # If no subdirs, treat data_dir itself as a project named 'default'
    if not projects:
        projects = ["."]

    for project_name in projects:
        project_path = os.path.join(data_dir, project_name)
        display_name = project_name if project_name != "." else "default"
        
        indexing_logger.info(f"Indexing project: {display_name}...")
        
        for root, dirs, files in os.walk(project_path):
            # Prune forbidden directories
            dirs[:] = [d for d in dirs if d.lower() not in FORBIDDEN_FOLDERS]
            
            for file in files:
                file_path = os.path.join(root, file)
                ext = os.path.splitext(file)[1].lower()
                
                # Special case for pom.xml which has no extension but is important
                is_pom = file.lower() == "pom.xml"
                
                if ext in EXTENSIONS or is_pom:
                    try:
                        # Use SafeTextLoader logic inline here for simplicity and control
                        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                            content = f.read()
                            if content.strip():
                                # Normalize extension for preprocessor (e.g. pom.xml -> xml)
                                proc_ext = ext if ext else (".xml" if is_pom else "")
                                content = preprocess_code(content, proc_ext)
                                all_docs.append(Document(
                                    page_content=content,
                                    metadata={
                                        "source": os.path.relpath(file_path, data_dir),
                                        "project": display_name,
                                        "filename": file,
                                        "extension": ext or "xml" # pom.xml has no ext
                                    }
                                ))
                    except Exception as e:
                        indexing_logger.error(f"Error loading {file_path}: {e}")

    return all_docs


def chunk_documents(docs: List[Document]) -> List[Document]:
    from langchain_text_splitters import Language
    
    java_splitter = RecursiveCharacterTextSplitter.from_language(
        language=Language.JAVA,
        chunk_size=1200, # Code often needs slightly larger chunks for context
        chunk_overlap=200,
    )
    
    default_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=150,
    )

    python_splitter = RecursiveCharacterTextSplitter.from_language(
        language=Language.PYTHON,
        chunk_size=1000,
        chunk_overlap=150,
    )

    split_docs = []
    for doc in docs:
        ext = doc.metadata.get("extension")
        if ext == ".java":
            split_docs.extend(java_splitter.split_documents([doc]))
        elif ext == ".py":
            split_docs.extend(python_splitter.split_documents([doc]))
        else:
            split_docs.extend(default_splitter.split_documents([doc]))
            
    return split_docs


def build_vector_store() -> Chroma:
    docs = load_documents(settings.resolved_data_dir)
    if not docs:
        indexing_logger.warning(f"No documents found in {settings.resolved_data_dir}")
        raise SystemExit(0)

    embeddings = make_embedder()
    
    # Initialize the vector store
    vectordb = Chroma(
        persist_directory=settings.active_chroma_dir,
        embedding_function=embeddings,
        collection_name="local-rag",
    )

    # --- INCREMENTAL INDEXING LOGIC ---
    # Get all sources already in the DB to avoid re-embedding
    try:
        existing_data = vectordb.get()
        existing_sources = set()
        if existing_data and 'metadatas' in existing_data:
            for meta in existing_data['metadatas']:
                if 'source' in meta:
                    existing_sources.add(meta['source'])
        
        indexing_logger.info(f"Found {len(existing_sources)} files already in database.")
        
        # Filter out docs that are already indexed
        new_docs = [d for d in docs if d.metadata.get('source') not in existing_sources]
        
        if not new_docs:
            indexing_logger.info("All files are already up to date. Nothing to index.")
            return vectordb
            
        indexing_logger.info(f"Indexing {len(new_docs)} new/updated files...")
        chunks = chunk_documents(new_docs)
    except Exception as e:
        indexing_logger.warning(f"Could not perform incremental check: {e}. Falling back to full indexing.")
        chunks = chunk_documents(docs)

    indexing_logger.info(f"Loaded documents -> {len(chunks)} chunks to process.")

    # --- BATCHED PROCESSING ---
    # Batch size of 50 stays safely under the 20,000 tokens-per-request limit
    # 1100 chunks = 22 requests (well under the 1000 daily limit)
    # To maximize the 1,000 daily requests (RPD), we use a larger batch size.
    # 45 chunks stays safely under the 20,480 token per-request size limit.
    # Increase batch size to 100 for better efficiency (roughly 25k tokens per request)
    batch_size = 100
    import time

    indexing_logger.info(f"Adding documents in batches of {batch_size} (saves daily quota)...")
    for i in range(0, len(chunks), batch_size):
        batch = chunks[i : i + batch_size]
        try:
            vectordb.add_documents(batch)
            indexing_logger.info(f"Processed {min(i + batch_size, len(chunks))}/{len(chunks)}")
            
            if settings.embedding_provider == "google":
                # 4.1s delay allows ~14.6 RPM, staying just under the 15 RPM Free Tier limit
                time.sleep(4.1) 
        except Exception as e:
            error_str = str(e)
            indexing_logger.error(f"Error processing batch starting at index {i}: {error_str}")
            
            # Check for Daily Quota exhaustion (RPD)
            if "EmbedContentRequestsPerDay" in error_str or "limit: 1000" in error_str:
                indexing_logger.critical("EXCEEDED DAILY GOOGLE API QUOTA (1,000 requests). Stopping indexer.")
                print("\nCRITICAL: Daily Google API Limit reached (1,000 requests).")
                print("The quota resets at midnight Pacific Time. Incremental indexing will resume tomorrow.")
                return vectordb

            # Robust retry for transient errors or TPM/RPM spikes
            wait_time = 180 
            if "RESOURCE_EXHAUSTED" in error_str:
                indexing_logger.info(f"Rate limit exhausted. Cooling down for {wait_time}s...")
            else:
                indexing_logger.info(f"Unexpected error. Cooling down for {wait_time}s...")
            
            time.sleep(wait_time)
            try:
                vectordb.add_documents(batch)
            except Exception as e2:
                indexing_logger.error(f"Second attempt failed: {e2}. Skipping batch.")

    vectordb.persist()
    indexing_logger.info("Indexing complete!")
    return vectordb
