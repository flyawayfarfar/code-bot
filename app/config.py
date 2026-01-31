# app/config.py

from pydantic_settings import BaseSettings
import os

class Settings(BaseSettings):
    # Switches
    # use_local_llm:         bool = True           # DEPRECATED/UNUSED (logic moved to main.py)
    # use_local_embeddings:  bool = True           # DEPRECATED
    
    # "ollama", "openai", "google"
    embedding_provider:    str  = "google"

    # Data / Vector DB
    data_dir:              str  = "data"
    chroma_dir:            str  = "chroma_db"
    chroma_dir_local: str = "chroma_db_ollama"
    chroma_dir_openai: str = "chroma_db_openai"
    chroma_dir_google: str = "chroma_googleai"
    k_neighbors:           int  = 8

    # Ollama
    ollama_base_url:       str  = "http://localhost:11434"
    ollama_model:          str  = "llama3.1:8b"
    ollama_embed_model:    str  = "nomic-embed-text"   # good local embedder
    local_embedding_model: str  = "nomic-embed-text"   # alias for older code

    # (optional) OpenAI fallback
    openai_api_key:        str | None = None

    # (optional) Google fallback
    google_api_key:        str | None = None
    google_model:          str = "gemini-2.5-flash-lite"

    class Config:
        env_file = ".env"
        extra = "ignore"

    def _resolve_path(self, path: str) -> str:
        """Helper to resolve absolute and relative paths."""
        if os.path.isabs(path):
            return path
        else:
            # Relative to project root (where main.py is located)
            return os.path.join(os.path.dirname(os.path.dirname(__file__)), path)

    @property
    def resolved_data_dir(self) -> str:
        """Resolve data directory path, supporting both absolute and relative paths."""
        return self._resolve_path(self.data_dir)

    @property
    def resolved_chroma_dir_local(self) -> str:
        return self._resolve_path(self.chroma_dir_local)

    @property
    def resolved_chroma_dir_openai(self) -> str:
        return self._resolve_path(self.chroma_dir_openai)

    @property
    def resolved_chroma_dir_google(self) -> str:
        return self._resolve_path(self.chroma_dir_google)

    @property
    def active_chroma_dir(self) -> str:
        if self.embedding_provider == "google":
            return self.resolved_chroma_dir_google
        if self.embedding_provider == "openai":
            return self.resolved_chroma_dir_openai
        return self.resolved_chroma_dir_local
    
settings = Settings()
